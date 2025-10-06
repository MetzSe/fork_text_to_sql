import asyncio
import json
import logging
import os
from abc import abstractmethod
from datetime import datetime, timezone
from functools import cache
from typing import (
    Optional,
    Dict,
    Type,
    Callable,
    TypeVar,
    ParamSpec,
    Awaitable,
    Union,
    Any,
)
from pydantic import BaseModel
from functools import wraps
import google.genai as genai

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

import boto3
import requests
import tiktoken
from openai import AsyncAzureOpenAI
from sqlalchemy import create_engine

from ascent_ai.config.settings import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


@cache
def _import_vertexai_generative_model():
    from vertexai.preview.generative_models import GenerativeModel

    return GenerativeModel


def make_generative_model(*args, **kwargs):
    cls = _import_vertexai_generative_model()
    return cls(*args, **kwargs)


class RetryError(Exception):
    """Base class for retry-related errors"""

    pass


class AssistantResponseError(Exception):
    """Raised when an assistant fails to generate a valid response after multiple attempts"""

    pass


class EmptyResponseError(Exception):
    """Exception raised when an empty response is received from the model."""
    pass


class InvalidJsonResponseError(Exception):
    """Exception raised when the model returns invalid JSON that doesn't match the expected schema."""
    pass


class RetryMixin:
    """Base class providing retry functionality"""

    def __init__(
        self, max_retries: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor

    @staticmethod
    def calculate_sleep_time(
        attempt: int, base_delay: float, backoff_factor: float
    ) -> float:
        return base_delay * (backoff_factor**attempt)

    async def handle_retry(
        self,
        err: Exception,
        attempt: int,
        max_retries: int,
        base_delay: float,
        backoff_factor: float,
    ) -> bool:
        """
        Default retry handling logic with exponential backoff.

        Args:
            err: The exception that triggered the retry
            attempt: Current attempt number (0-based)
            max_retries: Maximum number of retries
            base_delay: Initial delay between retries
            backoff_factor: Multiplicative factor for backoff

        Returns:
            bool: True if should retry, False otherwise
        """
        if attempt < max_retries - 1:
            sleep_time = self.calculate_sleep_time(attempt, base_delay, backoff_factor)
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {str(err)}. "
                f"Retrying in {sleep_time:.2f} seconds..."
            )
            await asyncio.sleep(sleep_time)
            return True
        logger.error(
            f"All {max_retries} retry attempts failed. " f"Last error: {str(err)}"
        )
        return False


def async_retry(
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    backoff_factor: Optional[float] = None,
    retry_on: Optional[Union[type[Exception], tuple[type[Exception], ...]]] = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Advanced async retry decorator with configurable parameters and error handling.

    Args:
        max_retries: Maximum number of retry attempts. If None, uses instance value.
        base_delay: Initial delay between retries. If None, uses instance value.
        backoff_factor: Multiplicative factor for backoff. If None, uses instance value.
        retry_on: Specific exception types to retry on. If None, retries on all exceptions.
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> T:
            # Use instance values if decorator params are None
            actual_max_retries = max_retries or getattr(self, "max_retries", 3)
            actual_base_delay = base_delay or getattr(self, "base_delay", 1.0)
            actual_backoff_factor = backoff_factor or getattr(
                self, "backoff_factor", 2.0
            )

            last_exception = None
            for attempt in range(actual_max_retries):
                try:
                    return await func(self, *args, **kwargs)
                except Exception as err:
                    # Check if we should retry this specific exception
                    if retry_on and not isinstance(err, retry_on):
                        raise

                    last_exception = err

                    # Custom retry handling if available
                    if hasattr(self, "handle_retry"):
                        should_retry = await self.handle_retry(
                            err,
                            attempt,
                            actual_max_retries,
                            actual_base_delay,
                            actual_backoff_factor,
                        )
                        if not should_retry:
                            logger.warning(
                                f"Retry handler decided to stop retrying after attempt {attempt + 1}"
                            )
                            raise

                    # Standard retry handling
                    if attempt < actual_max_retries - 1:
                        delay = actual_base_delay * (actual_backoff_factor**attempt)
                        logger.info(
                            f"Attempt {attempt + 1}/{actual_max_retries} failed: {str(err)}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {actual_max_retries} retry attempts failed. "
                            f"Last error: {str(err)}"
                        )
                    continue

            raise last_exception or RetryError("Unexpected end of retry loop")

        return wrapper

    return decorator


class BaseAssistant(RetryMixin):
    def __init__(self, system_message: Optional[dict] = None):
        super().__init__(max_retries=5, base_delay=1, backoff_factor=1.5)

        self.system_message = system_message or {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        self.conversation = [self.system_message]

    @abstractmethod
    async def get_response(
        self,
        prompt: Optional[str] = None,
        temperature=0,
        json_format: Optional[bool] = False,
        quiet: Optional[bool] = False,
        response_schema: Optional[Type[BaseModel]] = None,
    ):
        pass

    def reset_conversation(self):
        self.conversation = [self.system_message]

    def add_message(self, role, message):
        self.conversation.append({"role": role, "content": message})


# Define a registry for assistant creators
assistant_registry: Dict[str, Type[BaseAssistant]] = {}


# A decorator function for registering assistant creators
def register_assistant(assistant_type: str):
    def decorator(cls: Type[BaseAssistant]):
        assistant_registry[assistant_type] = cls
        return cls

    return decorator


@register_assistant("gpt")
class GPTAssistant(BaseAssistant):
    model_name: str
    api_key: str
    api_base: str
    api_version: str
    max_response_tokens: int
    client: AsyncAzureOpenAI

    OPENAI_API_VERSION = "2024-02-01"

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        # Only use settings if no value is provided
        model_name = model_name or settings.MODEL_NAME
        api_key = api_key or settings.OPENAI_API_KEY_GPT4O
        api_base = api_base or settings.OPENAI_API_BASE
        api_version = api_version or self.OPENAI_API_VERSION

        super().__init__()

        self.model_name = model_name or settings.MODEL_NAME
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.api_base = api_base or settings.OPENAI_API_BASE
        self.api_version = api_version or settings.OPENAI_API_VERSION

        # If you need to override the default system message
        self.system_message = {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        self.conversation = [
            self.system_message
        ]  # Reset conversation with new system message

        self.max_response_tokens = 4096
        self.token_limit = 8192 * 4
        self.conversation = [self.system_message]
        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.api_base,
        )

    @staticmethod
    def num_tokens_from_messages(messages):
        encoding = tiktoken.encoding_for_model("gpt-4-32k")
        num_tokens = 0
        for message in messages:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    def add_message(self, role, message):
        self.conversation.append({"role": role, "content": message})
        self.manage_conversation_length()

    def manage_conversation_length(self):
        conv_history_tokens = self.num_tokens_from_messages(self.conversation)

        while conv_history_tokens + self.max_response_tokens >= self.token_limit:
            del self.conversation[1]
            conv_history_tokens = self.num_tokens_from_messages(self.conversation)

    @async_retry(retry_on=(ConnectionError, TimeoutError))
    async def get_response(
        self,
        prompt: Optional[str] = None,
        temperature=0,
        json_format: Optional[bool] = False,
    ):
        """
        Sends a prompt to Gpt and retrieves the response.

        This method formats the prompt as required, sends it to Gpt, and handles retries in case of failures.

        Parameters:
        - prompt (Optional[str]): The input prompt to send to Gpt. If None, the existing conversation is used.
        - json_format (Optional[bool]): A flag indicating whether the response should be formatted as JSON. If True,
          the response will include a system message and be structured as a JSON object.

        Returns:
        - str: The content of the message from Gpt's response.

        Note:
        - If `json_format` is True, the response will be a JSON object; otherwise, it will be a plain text string.
        """
        messages = self.prepare_messages(prompt, json_format)
        response = await self.make_api_call(messages, temperature, json_format)
        return self.process_response(response, prompt)

    def prepare_messages(self, prompt, json_format):
        if json_format:
            return [
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to output JSON.",
                },
                {"role": "user", "content": prompt},
            ]
        return (
            [{"role": "user", "content": prompt}]
            if prompt is not None
            else self.conversation
        )

    async def make_api_call(self, messages, temperature, json_format):
        """Template method that uses _get_api_params"""
        params = self._get_api_params(messages, temperature, json_format)
        return await self.client.chat.completions.create(**params)

    def _get_api_params(self, messages, temperature, json_format):
        """Returns the API parameters for standard GPT models"""
        params = {
            "model": self.model_name,
            "temperature": temperature,
            "messages": messages,
            "max_tokens": self.max_response_tokens,
            "timeout": 600,
        }
        if json_format:
            params["response_format"] = {"type": "json_object"}
        return params

    def process_response(self, response, prompt):
        if prompt is None:
            self.add_message(
                role="assistant", message=response.choices[0].message.content
            )
        return response.choices[0].message.content


@register_assistant("gpt-o1")
class GPTo1Assistant(GPTAssistant):
    """
    GPTo1Assistant inherits from GPTAssistant to handle the OpenAI O1 preview model.
    The main difference is in the API parameters - O1 uses 'max_completion_tokens'
    instead of 'max_tokens' which is used by other GPT models.
    """

    DEFAULT_MODEL = None
    DEFAULT_API_VERSION = None
    OPENAI_API_BASE = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        # Use class defaults before falling back to settings
        model_name = model_name or self.DEFAULT_MODEL
        api_version = api_version or self.DEFAULT_API_VERSION
        api_base = api_base or self.OPENAI_API_BASE
        api_key = api_key or settings.OPENAI_API_KEY_O1

        # Initialize the parent class with all necessary parameters
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
        )

    # TO DO: clean up temperature since it is not needed
    def _get_api_params(self, messages, temperature, json_format):
        """Returns the API parameters for O1 model"""
        params = {
            "model": self.model_name,
            # "temperature": temperature,   #not available
            "messages": messages,
            "max_completion_tokens": self.max_response_tokens,  # Only difference
            "timeout": 600,
        }
        if json_format:
            params["response_format"] = {"type": "json_object"}
        return params

    def prepare_messages(self, prompt, json_format):
        """Override to handle the fact that O1 doesn't support system messages"""
        if json_format:
            # For JSON format, include the instruction in the user message
            return [
                {
                    "role": "user",
                    "content": "You are a helpful assistant designed to output JSON. "
                    + prompt,
                }
            ]

        if prompt is not None:
            # For new prompts, combine system message content with the prompt
            system_content = self.system_message["content"]
            return [{"role": "user", "content": f"{system_content}\n\n{prompt}"}]

        # For conversation history, filter out system messages and convert them
        messages = []
        for msg in self.conversation:
            if msg["role"] == "system":
                # Convert system message to user message
                messages.append({"role": "user", "content": msg["content"]})
            else:
                messages.append(msg)
        return messages


@register_assistant("gemini")
class GeminiAssistant(BaseAssistant):
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        max_output_tokens: int = 4096,
        temperature: float = 0,
        top_p: float = 0.8,
    ):
        super().__init__()

        # Configure API key
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not found")

        # Initialize the client with API key
        self.client = genai.Client(api_key=self.api_key)

        # Initialize model and parameters
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Initialize conversation history
        self.conversation = []

    @async_retry(retry_on=(ConnectionError, TimeoutError, EmptyResponseError, InvalidJsonResponseError))
    async def get_response(
            self,
            prompt: Optional[str] = None,
            temperature: Optional[float] = None,
            json_format: Optional[bool] = False,
            quiet: Optional[bool] = False,
            response_schema: Optional[Type[BaseModel]] = None,
            **kwargs,
    ) -> Union[str, Dict[str, Any], BaseModel]:
        """
        Get a response from the Gemini model.

        Args:
            prompt: The input prompt
            temperature: Temperature for response generation
            json_format: Whether to return response in JSON format
            quiet: Return log is response correctly received
            response_schema: Optional Pydantic model to structure the response
            **kwargs: Additional parameters

        Returns:
            Union[str, Dict[str, Any], BaseModel]: The generated response
        """
        try:
            if prompt is None:
                prompt = str(self.conversation)
                logger.info("No prompt passed, using conversation history.")

            # Add JSON format instruction if needed
            if json_format or response_schema:
                # Enhance the prompt with more specific JSON formatting instructions
                json_instruction = (
                    "Your response must be valid, properly formatted JSON that can be parsed by Python's json.loads(). "
                    "Ensure all quotes are properly escaped and all brackets/braces are balanced. "
                    "Do not include any explanatory text outside the JSON structure."
                )
                prompt = f"{json_instruction}\n\n{prompt}"

            # Prepare config dictionary with generation parameters
            config = {
                "max_output_tokens": self.max_output_tokens,
                "temperature": (
                    temperature if temperature is not None else self.temperature
                ),
                "top_p": self.top_p,
            }

            # Add response schema to config if present
            if response_schema:
                # Convert Pydantic model to JSON schema
                schema_dict = response_schema.model_json_schema()
                config["response_mime_type"] = "application/json"
                config["response_schema"] = schema_dict

            # Generate response with inline generation config
            # response = self.client.models.generate_content(
            #     model=self.model_name, contents=prompt, config=config
            # )
            # async version
            response = await self.client.aio.models.generate_content(
                model=self.model_name, contents=prompt, config=config
            )

            # Check for empty response and raise custom exception to trigger retry
            if not response.text or response.text.strip() == "":
                raise EmptyResponseError("Empty response received from Gemini")

            # Store in conversation history
            self.add_message("user", prompt)
            self.add_message("assistant", response.text)

            if not quiet:
                logger.info(
                    f"Successfully generated response with model: gemini, "
                    f"timestamp: {datetime.now(timezone.utc).strftime('%Y.%m.%d %H:%M')}"
                )

            # Process response based on format requirements
            if response_schema:
                try:
                    # Try to parse the JSON response
                    parsed_data = json.loads(response.text)
                    # Try to validate against the schema
                    return response_schema(**parsed_data)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse response as schema: {e}")
                    # Instead of returning the raw text, raise an exception to trigger retry
                    raise InvalidJsonResponseError(f"Invalid JSON response for schema: {e}")
            elif json_format:
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse response as JSON: {e}")
                    raise InvalidJsonResponseError(f"Invalid JSON format: {e}")

            return response.text

        except EmptyResponseError as e:
            # Log the empty response error before re-raising
            logger.warning(f"Received empty response from Gemini API, triggering retry: {str(e)}")
            # Re-raise to be caught by the retry decorator
            raise

        except InvalidJsonResponseError as e:
            # Log the JSON parsing error before re-raising
            logger.warning(f"Received malformed JSON from Gemini API, triggering retry: {str(e)}")
            # Re-raise to be caught by the retry decorator
            raise

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation = []

    def add_message(self, role: str, message: str):
        """Add a message to the conversation history"""
        self.conversation.append({"role": role, "content": message})

@register_assistant("claude_sonnet")
class ClaudeSonnetAssistant(BaseAssistant):
    DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.0,
        top_p: float = 0.9,
        region_name: str = "us-east-1",
    ):
        # Initialize a boto3 client for the AWS Bedrock.
        self._client = None
        self.region_name = region_name
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_message = {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        self.conversation = [self.system_message]

        super().__init__()

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self.region_name)
        return self._client

    def create_request(self, prompt):
        # Create a request dictionary.
        request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "system": "You are a helpful assistant",
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
        }
        return request

    def invoke_model(self, request):
        # Send the request to aws bedrock and return the response after parsing the JSON response body.
        response = self.client.invoke_model(
            body=json.dumps(request), modelId=self.model_name
        )
        model_response = json.loads(response.get("body").read())
        return model_response.get("content")[0].get("text")

    @async_retry(retry_on=(ConnectionError, TimeoutError))
    async def get_response(self, prompt: Optional[str] = None, **kwargs):
        if prompt is None:
            prompt = str(self.conversation)
            logger.info("No prompt passed, using conversation.")

        if kwargs:
            logger.warning(
                f"The following kwargs were passed but are not used: {', '.join(kwargs.keys())}"
            )

        # Generate a response from the aws bedrock.
        request = self.create_request(prompt)
        response_text = self.invoke_model(request)

        logger.info(
            f"Successful {self.model_name} response model: {self.model_name}, "
            f"utc-timestamp: {datetime.now(timezone.utc).strftime('%Y.%m.%d %H:%M')}"
        )
        logger.debug(f"message:{str(prompt)}, response-content: {response_text}")

        return response_text



@register_assistant("bedrock_mistral")
class BedrockMistralAssistant(BaseAssistant):
    DEFAULT_MODEL = "mistral.mistral-large-2402-v1:0"

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_tokens: int = 200,
        temperature: float = 0.5,
        top_p: float = 0.9,
        top_k: int = 50,
        region_name: str = "us-east-1",
        system_message: Optional[dict] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ):
        super().__init__(system_message)
        self._client = None
        self.region_name = region_name
        self.model_id = model_name or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        # Update retry parameters
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self.region_name)
        return self._client

    @staticmethod
    def create_prompt(user_message: str) -> str:
        prompt = f"<s>[INST] {user_message} [/INST]"
        return prompt

    def create_request(
        self, prompt, max_tokens=None, temperature=None, top_p=None, top_k=None
    ):
        request = {
            "prompt": prompt,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p,
            "top_k": top_k or self.top_k,
        }
        return request

    @async_retry(retry_on=(Exception,))
    async def invoke_model(self, request):
        try:
            response = self.client.invoke_model(
                body=json.dumps(request),
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response["body"].read())
            return response_body.get("generation", "")
        except Exception as e:
            logger.error(f"Error invoking model: {str(e)}")
            raise

    @async_retry(retry_on=(ConnectionError, TimeoutError))
    async def get_response(
        self,
        prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        json_format: Optional[bool] = False,
        **kwargs,
    ):
        if prompt is None:
            prompt = str(self.conversation)
            logger.info("No prompt passed, using conversation.")

        if kwargs:
            logger.warning(
                f"The following kwargs were passed but are not used: {', '.join(kwargs.keys())}"
            )

        formatted_prompt = self.create_prompt(prompt)
        request = self.create_request(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        response_text = await self.invoke_model(request)

        logger.info(
            f"Successful {self.model_id} response model: {self.model_id}, "
            f"utc-timestamp: {datetime.now(timezone.utc).strftime('%Y.%m.%d %H:%M')}"
        )
        logger.debug(f"message:{str(prompt)}, response-content: {response_text}")

        if json_format:
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse response as JSON")
                return response_text

        return response_text


def create_assistant(assistant_type: str, **kwargs) -> Optional[BaseAssistant]:
    """
    :param assistant_type: Either gpt, gemini, linguist, mistral
    :param kwargs: Optional parameters model_name, api_key, api_base, api_version for different model usage from gpt
    :return:
    """
    assistant_cls = assistant_registry.get(assistant_type)
    if not assistant_cls:
        logging.exception(f"Assistant type '{assistant_type}' is not registered.")
        return None
    return assistant_cls(**kwargs)


def create_snowflake_engine(url: str):
    """
    Create a Snowflake engine using the given URL.

    Args:
    url (str): a Snowflake URL in the following format
               snowflake://user:password@account/database/schema?warehouse=warehouse&role=role

    Returns:
    engine: SQLAlchemy engine instance
    """
    engine = create_engine(url)

    # Test the connection
    try:
        connection = engine.connect()
        connection.close()
        logger.info("Successfully connected to the database.")
    except Exception as e:
        logger.exception("Error occurred:", e)
        raise

    return engine

