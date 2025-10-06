import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import pandas as pd
from typing_extensions import TypedDict

from ascent_ai.models.inference import prompts
from ascent_ai.models.inference.rag import RAGProcessor

from ascent_ai.schemas.data_definitions import Query
from ascent_ai.db.snowflake_session import get_db
from ascent_ai.data.processing.encoder import ExtendedEncoder

from ascent_ai.models.inference.assistants import create_assistant
from ascent_ai.utils.mixing_temp import get_text_sql_template_for_rag, prepare_prediction, add_messages_to_assistant
from ascent_ai.schemas.data_definitions import CohortMetadata

logger = logging.getLogger(__name__)


class SQLProcessor(Protocol):
    """Protocol defining the interface for SQL processors"""

    def parse_sql_from_response(self, response: str) -> str:
        """Parse SQL from a response string"""
        ...



@dataclass
class QueryContext:
    """Context for query generation process"""

    user_input: Optional[str] = None
    query_id: str = None
    rag_agent: Optional[RAGProcessor] = None
    sql_processor: Optional[SQLProcessor] = None
    preferred_coding_system: Optional[str] = None
    cohort_metadata: Optional[CohortMetadata] = None


@dataclass
class QueryResult:
    """Structured result from query processing"""

    ai_answer: str
    question_masked: str
    initial_prompt: str
    rag_prompt_add_on: str
    df_recs_list_out: Optional[pd.DataFrame]
    explanation: Optional[str] = None


class CriteriaDict(TypedDict):
    """Type definition for criteria dictionary"""

    include: List[str]
    exclude: List[str]
    masked_text: Optional[Dict[str, List[str]]]


class CriteriaParser:
    @staticmethod
    def extract_criteria(criteria_dict: Union[Dict, str]) -> List:
        """Extract criteria from dictionary or string"""
        if isinstance(criteria_dict, str):
            criteria_dict = CriteriaParser.parse_string_to_dict(criteria_dict)

        try:
            return criteria_dict["include"] + criteria_dict["exclude"]
        except KeyError:
            return criteria_dict["masked_text"]["include"] + criteria_dict["masked_text"]["exclude"]

    @staticmethod
    def parse_string_to_dict(input_str: str) -> Optional[Dict]:
        """Convert string representation to dictionary"""
        try:
            match = re.search(r"\{.*}", input_str, re.DOTALL)
            if not match:
                raise ValueError("No dictionary-like structure found in string")

            dict_string = match.group(0)

            try:
                logger.debug(f"Try to parse {dict_string=} as JSON")
                return json.loads(dict_string)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse {dict_string=} as JSON: {e}")
                raise ValueError(f"Invalid JSON format in dictionary string: {e}")

        except (SyntaxError, ValueError) as e:
            logger.error(f"Error while extracting dictionary from {input_str!r}")
            logger.exception(e)
            return None


@dataclass
class ModelHandler:
    """Handles interactions with the AI model"""

    rag_agent: RAGProcessor
    cohort_metadata: Optional[CohortMetadata] = None

    async def prepare_model_call(self, user_input: str, mask_assistant=None, rag_random=False, drop_first=False) -> Tuple[str, str, pd.DataFrame, str]:
        """Prepare model call with necessary context"""
        if not mask_assistant:
            # mask_assistant = self.rag_agent.default_assistant
            mask_assistant = create_assistant(assistant_type="claude_sonnet",
                                              model_name="us.anthropic.claude-sonnet-4-20250514-v1:0")

        question_masked, question = await self.rag_agent.querylib.get_masked_question(question=user_input,
                                                                                      assistant=mask_assistant)

        initial_prompt = prepare_prediction(question, prompt=prompts.prompt_gpt)

        if self.cohort_metadata:
            initial_prompt = self._add_cohort_prompt(initial_prompt)

        text_sql_template, df_recs_list_out = await get_text_sql_template_for_rag(question_masked=question_masked, rag=self.rag_agent,
                                                                                  rag_random=rag_random, drop_first=drop_first
                                                                                  )

        add_messages_to_assistant([initial_prompt, text_sql_template], self.rag_agent.default_assistant)

        return initial_prompt, text_sql_template, df_recs_list_out, question_masked

    def _add_cohort_prompt(self, prompt: str) -> str:
        """Add cohort-specific information to prompt"""
        return re.sub(
            r"^#\s*question\s*:?\s*$",
            f"{prompts.cohort_prompt_addon(self.cohort_metadata)}\n# Question:",
            prompt,
            flags=re.IGNORECASE | re.MULTILINE,
            count=1,
        )

    async def get_response(self) -> str:
        """Get model response and handle incorrect patterns"""
        response = await self.rag_agent.default_assistant.get_response()

        if re.search(r"concept_name\s*=", response):
            correction_message = """
            The generated SQL query contains 'concept_name = ***', which seems incorrect.
            Please review the query generation steps and provide a revised answer.
            """
            add_messages_to_assistant([correction_message], self.rag_agent.default_assistant)
            response = await self.rag_agent.default_assistant.get_response()

        logger.info("Model response generation completed")
        return response


class QueryProcessor:
    """Main query processing orchestrator"""

    def __init__(self, context: QueryContext, skip_db_upload: bool = False):
        self.context = context
        self.model_handler = ModelHandler(rag_agent=context.rag_agent, cohort_metadata=context.cohort_metadata)
        self.skip_db_upload = skip_db_upload

    async def process_query(self, drop_first=False, rag_random=False) -> QueryResult:
        """Process query and return structured result"""

        initial_prompt, text_sql_template, df_recs_list_out, question_masked = await self.model_handler.prepare_model_call(self.context.user_input,
                                                                                                                           drop_first=drop_first,
                                                                                                                           rag_random=rag_random)

        ai_answer = await self.model_handler.get_response()
        query_template_pred = self.context.sql_processor.parse_sql_from_response(ai_answer)

        query = {
            "id": self.context.query_id,
            "gpt_answer": query_template_pred,
            "initial_prompt": initial_prompt,
            "text_sql_template": text_sql_template,
            "df_recs_list_out": json.dumps(df_recs_list_out.to_dict("records"), cls=ExtendedEncoder),
            "question_masked": question_masked,
        }

        if self.context.cohort_metadata:
            query["cohort_id"] = self.context.cohort_metadata.id

        session_data = Query(**query)

        df_recs_list = self._prepare_dataframe(session_data.df_recs_list_out)

        return QueryResult(
            ai_answer=session_data.gpt_answer, question_masked=session_data.question_masked,
            initial_prompt=session_data.initial_prompt,
            rag_prompt_add_on=session_data.text_sql_template,
            df_recs_list_out=df_recs_list,
        )

    @staticmethod
    def _prepare_dataframe(df_recs_list_out: Optional[str]) -> Optional[pd.DataFrame]:
        """Prepare DataFrame from JSON string"""
        if not df_recs_list_out:
            return None

        df_recs_list = json.loads(df_recs_list_out)
        if not df_recs_list:
            return None

        df = pd.DataFrame(df_recs_list)
        return df if not df.empty else None


async def process_single_database(
        database_name: str,
        ai_answer: str,
        med_sql_processor,
        preferred_coding_system,
        rag_agent,
        schema_name: Optional[str] = None
) -> tuple[str, str]:
    """Process a single database query"""
    try:
        async with get_db(database_name, schema_name) as snowflake_session:
            query_filled = await med_sql_processor.post_process_sql_query(
                sql_text=ai_answer,
                preferred_coding_system=preferred_coding_system,
                rag_agent=rag_agent,
                db_session=snowflake_session,
            )
            return database_name, query_filled
    except Exception as e:
        # Log the error and return None or handle as needed
        return database_name, f"Error processing database {database_name}: {str(e)}"


async def generate_sql_query(user_input: str, return_filled_query: bool = True, skip_db_upload: bool = False,
                             drop_first: bool = False,
                             rag_random: bool = False, cohort_metadata: CohortMetadata = None, **kwargs) -> Dict[
    str, Any]:
    """Main entry point for generating template queries with parallel database processing"""
    # Extract kwargs
    med_sql_processor = kwargs.get("med_sql_processor")
    preferred_coding_system = kwargs.get("preferred_coding_system")
    rag_agent = kwargs.get("rag_agent")
    selected_databases = kwargs.get("selected_databases")

    # Initialize context and process query
    context = QueryContext(
        user_input=user_input,
        query_id=kwargs.get("query_id"),
        rag_agent=rag_agent,
        sql_processor=med_sql_processor,
        preferred_coding_system=preferred_coding_system,
        cohort_metadata=cohort_metadata,
    )
    processor = QueryProcessor(context, skip_db_upload=skip_db_upload)
    result = await processor.process_query(rag_random=rag_random, drop_first=drop_first)
    template_query = result.ai_answer

    if return_filled_query:
        # Process all databases in parallel
        tasks = [
            process_single_database(database_name, result.ai_answer, med_sql_processor, preferred_coding_system,
                                    rag_agent)
            for database_name in selected_databases
        ]

        # Gather results
        database_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert results to dictionary
        filled_queries = {db_name: query_result for db_name, query_result in database_results if
                          query_result is not None}
    else:
        # When not returning filled queries, just provide an empty dictionary
        filled_queries = {}

    return {
        "template_query": template_query,
        "filled_queries": filled_queries,
        "result": result
    }
