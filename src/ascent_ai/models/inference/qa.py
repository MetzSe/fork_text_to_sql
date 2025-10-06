import asyncio
import copy
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, List
import pandas as pd

from ascent_ai.models.inference.rag import RAGConfig, RAGProcessor
from ascent_ai.data.queries import result_query_builder
from ascent_ai.data.processing.sql_post_processor import MedicalSQLProcessor
from ascent_ai.data.processing.utils_processing import lowercase_placeholder_values
from ascent_ai.utils.mixing_temp import prepare_rwd_request
from ascent_ai.db.snowflake_session import get_db
from ascent_ai.external_calls import MedicalCoder
from ascent_ai.schemas.constants import CodingType
from ascent_ai.models.inference.sql_explainer import SQLExplainer
from ascent_ai.schemas.data_definitions import CohortMetadata


class QuestionAnsweringSystem:
    def __init__(
            self,
            base_dir: Path,
            log_folder: Path,
            querylib_file: Path,
            assistant_type: str = "gpt",
            model_name: str | None = None,
            snowflake_database: str = None,
            snowflake_database_schema: str = None,
            settings=None
    ):
        self.base_dir = base_dir
        self.log_folder = log_folder
        self.querylib_file = querylib_file
        self.assistant_type = assistant_type
        self.model_name = model_name
        self.snowflake_database = snowflake_database
        self.snowflake_database_schema = snowflake_database_schema
        self.settings = settings

        self.config = RAGConfig(
            main_path=str(base_dir),
            log_folder=str(log_folder),
            assistant_type=assistant_type,
            model_name=model_name if model_name is not None else None
        )

        self.rag_agent = RAGProcessor(self.config)
        self.rag_agent.querylib_manager.load_querylib(querylib_file=str(querylib_file))

        self.logger = logging.getLogger(__name__)

        # Medical coder is not implemented - set recommender to None
        self.recommender = None
        self.logger.info("MedicalCoder not implemented - using None for recommender")

        self.med_sql_processor = MedicalSQLProcessor(
            assistant=self.rag_agent.default_assistant,
            recommender=self.recommender
        )

        self.sql_explainer = SQLExplainer()

    @classmethod
    async def initialize(cls, config: Dict[str, Any]) -> "QuestionAnsweringSystem":
        """
        Factory method to create and initialize a QuestionAnsweringSystem instance.

        Args:
            config: Configuration dictionary containing:
                - querylib_file: Path to query library
                - assistant: Dict with type and model
                - database: Snowflake database configuration
                - database_schema: Snowflake database schema configuration
                - settings: Settings object with Azure credentials
                - log_directory: Optional path for logs

        Returns:
            Initialized QuestionAnsweringSystem instance
        """
        base_dir = Path(__file__).resolve().parent.parent

        # Create a logs directory if not specified in config
        if "log_directory" in config:
            log_folder = Path(config["log_directory"])
        else:
            log_folder = base_dir / "logs"  # default location
            log_folder.mkdir(parents=True, exist_ok=True)

        if "database_schema" in config:
            snowflake_database_schema = config["database_schema"]
        else:
            snowflake_database_schema = None

        querylib_file = base_dir / config["querylib_file"]

        return cls(
            base_dir=base_dir,
            log_folder=log_folder,
            querylib_file=querylib_file,
            assistant_type=config["assistant"]["type"],
            model_name=config["assistant"]["model"],
            snowflake_database=config["database"],
            snowflake_database_schema=snowflake_database_schema,
            settings=config.get("settings")
        )

    # ==================== SINGLE QUESTION METHODS ====================
    
    async def generate_query_template(
            self,
            input_question: str,
            selected_databases: list[str] = ["OPTUM_CLAIMS_OMOP"],
            drop_first: Optional[bool] = False,
            cohort_metadata: Optional[CohortMetadata] = None
    ) -> Dict[str, Any]:
        """
        Generate the SQL query template from the input question.

        Args:
            input_question: The question to generate SQL for
            selected_databases: List of databases to consider
            drop_first: True if the first retrieved SQL from the query library should be dropped, False otherwise
                        Always use False when deploying, and use True when evaluating/benchmarking the models
            cohort_metadata: metadata of the table, applicable only if a cohort is being queried

        Returns:
            Dict containing the template query and generation metadata
        """
        try:
            sql_query_result = await result_query_builder.generate_sql_query(
                user_input=input_question,
                return_filled_query=False,
                drop_first=drop_first,
                med_sql_processor=self.med_sql_processor,
                rag_agent=self.rag_agent,
                selected_databases=selected_databases,
                skip_db_upload=True,
                cohort_metadata=cohort_metadata,
            )

            # Lowercase the placeholder values in the template query
            template_query = lowercase_placeholder_values(sql_query_result["template_query"])

            return {
                "query_template": template_query,
                "generation_metadata": sql_query_result.get("metadata", {})
            }
        except Exception as e:
            self.logger.error(f"Error generating query template: {str(e)}")
            raise

    async def post_process_query(
            self,
            query_template: str,
            preferred_coding_system: CodingType = CodingType.STANDARD_CODING,
            custom_code_mappings: Optional[Dict[Tuple[str, str], str]] = None
    ) -> str:
        """
        Post-process the query template to fill in codes and other specifics.

        Args:
            query_template: The template SQL query
            preferred_coding_system: The coding system to use
            custom_code_mappings: Optional dictionary mapping (entity_type, value) to code strings
                                 Example:         custom_code_mappings = {
            ('condition', 'dysphagia'): '4310996,4159140',
            ('condition', 'atopic dermatitis'): '1112807, 1112808',
            ('procedure', 'appendectomy'): '2211444,2211445'

        Returns:
            The processed SQL query ready for execution
        """
        try:
            async with get_db(self.snowflake_database, self.snowflake_database_schema) as db:
                query_filled = await self.med_sql_processor.post_process_sql_query(
                    query_template,
                    preferred_coding_system=preferred_coding_system,
                    rag_agent=self.rag_agent,
                    db_session=db,
                    custom_code_mappings=custom_code_mappings
                )
                return query_filled
        except Exception as e:
            self.logger.error(f"Error post-processing query: {str(e)}")
            raise

    async def execute_query(
            self,
            query_filled: str,
            query_template: str,
            input_question: str,
            preferred_coding_system: str = "Standard",
            max_retries: int = 5,
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Execute the processed SQL query and return results.

        Args:
            query_filled: The processed SQL query ready for execution
            query_template: The original template query
            input_question: The original question
            preferred_coding_system: The coding system used
            max_retries: Maximum number of query retry attempts

        Returns:
            Tuple of (DataFrame with results, RWD request object)
        """
        try:
            async with get_db(self.snowflake_database, self.snowflake_database_schema) as db:
                new_prompt = self.rag_agent.default_assistant.conversation

                rwd_request = prepare_rwd_request(
                    user_input=input_question,
                    query_filled_pred=query_filled,
                    query_template_pred=query_template,
                    preferred_coding_system=preferred_coding_system,
                    med_sql_processor=self.med_sql_processor,
                    rag=self.rag_agent,
                    prompt=new_prompt,
                )

                df = await rwd_request.run_query(
                    db=db,
                    max_retries=max_retries,
                    reset_conversation=False,
                )

                return df, rwd_request
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise

    async def generate_answer(
            self,
            rwd_request: Any,
            df: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Generate an answer based on the query results.

        Args:
            rwd_request: The RWD request object containing query context
            df: Optional DataFrame with query results

        Returns:
            Generated answer string
        """
        try:
            await rwd_request.get_answer(self.rag_agent.default_assistant_answers)
            return rwd_request.answer
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            raise

    async def answer_question(
            self,
            input_question: str,
            drop_first: Optional[bool] = True,
            preferred_coding_system: str = "Standard",
            generate_answer: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a question end-to-end and return results.

        Args:
            input_question: The question to be answered
            preferred_coding_system: The coding system to use
            drop_first: True if the first retrieved SQL from the query library should be dropped, False otherwise
                Always use False when deploying, and use True when evaluating/benchmarking the models
            generate_answer: Whether to generate a natural language answer

        Returns:
            Dict containing results and metadata
        """
        start_time = time.time()

        try:
            # Generate query template
            template_result = await self.generate_query_template(input_question=input_question, drop_first=drop_first)
            query_template = template_result["query_template"]

            # Post-process query
            query_filled = await self.post_process_query(
                query_template,
                preferred_coding_system,
            )

            # Execute query
            df, rwd_request = await self.execute_query(
                query_filled,
                query_template,
                input_question,
                preferred_coding_system
            )

            # Generate answer if requested
            answer = None
            if generate_answer:
                answer = await self.generate_answer(rwd_request, df)

            execution_time = time.time() - start_time

            results = {
                "question": input_question,
                "answer": answer,
                "sql_template": query_template,
                "sql_filled": query_filled,
                "data": df.to_dict() if df is not None else None,
                "execution_time": execution_time,
                "generation_metadata": template_result.get("generation_metadata", {})
            }

            self.log_results(results)
            return results

        except Exception as e:
            error_time = time.time() - start_time
            error_results = {
                "question": input_question,
                "error": str(e),
                "execution_time": error_time
            }
            self.logger.error(f"Error processing question: {str(e)}")
            return error_results

    async def get_query_explanation(self, sql_query: str, input_question: str = None) -> str:
        return await self.sql_explainer.get_explanation(sql_query, input_question)

    # ==================== MULTIPLE QUESTIONS METHODS ====================

    @classmethod
    async def create_qa_system_for_question(cls, config: Dict[str, Any]) -> "QuestionAnsweringSystem":
        """Create a fresh QA system instance for a single question to ensure clean conversation context."""
        return await cls.initialize(config)

        # In your qa.py file, inside the QuestionAnsweringSystem class

    def _create_isolated_worker(self) -> Tuple[RAGProcessor, MedicalSQLProcessor]:
        """
        Creates a lightweight, isolated set of "worker" components for a single parallel task.
        This performs a fast, in-memory copy and avoids all blocking I/O.
        """
        # This surgical copy pattern creates an isolated context without blocking.
        local_rag_agent = copy.copy(self.rag_agent)
        local_rag_agent.default_state = copy.copy(self.rag_agent.default_state)

        # Isolate both assistants to keep conversations separate
        local_rag_agent.default_state.assistant = copy.copy(self.rag_agent.default_state.assistant)
        local_rag_agent.default_state.assistant.conversation = []

        local_rag_agent.default_state.assistant_answers = copy.copy(self.rag_agent.default_state.assistant_answers)
        local_rag_agent.default_state.assistant_answers.conversation = []

        # Create a new processor that uses the isolated agent
        local_med_sql_processor = MedicalSQLProcessor(
            assistant=local_rag_agent.default_assistant,
            recommender=self.recommender
        )

        return local_rag_agent, local_med_sql_processor

    async def prime_client(self):
        """
        Makes a single, cheap API call to force the underlying Boto3 client
        to initialize. This should be called once before running concurrent tasks.
        """
        try:
            self.logger.info("Priming the API client for parallel execution...")
            # This assumes your assistant uses AWS Bedrock.
            # Find the Boto3 client. It might be on self.rag_agent.default_assistant.client or similar.
            # Let's assume it's a BedrockRuntime client.
            # NOTE: You may need to adjust the path to your actual client object.
            boto_client = self.rag_agent.default_assistant.client

            # The specific call doesn't matter, as long as it's a real, cheap API call.
            # Listing models is a perfect candidate.
            await asyncio.to_thread(boto_client.list_foundation_models, byProvider='Anthropic')

            self.logger.info("API client is primed and ready.")
        except Exception as e:
            self.logger.error(f"Could not prime the API client. Parallel performance may be degraded. Error: {e}")

    async def _process_single_template_generation(
            self,
            question: str,
            drop_first: bool,
            selected_databases: list[str],
            cohort_metadata: Optional[Any]
    ) -> Dict[str, Any]:
        """
        Creates an isolated context using the central worker factory and awaits
        the now non-blocking RAG process.
        """
        # 1. Get a fresh, isolated worker. This is now a single, clean call.
        local_rag_agent, local_med_sql_processor = self._create_isolated_worker()

        # 2. Use the worker to do the job. The rest of the logic is unchanged.
        sql_query_result = await result_query_builder.generate_sql_query(
            user_input=question,
            return_filled_query=False,
            drop_first=drop_first,
            med_sql_processor=local_med_sql_processor,
            rag_agent=local_rag_agent,
            selected_databases=selected_databases,
            skip_db_upload=True,
            cohort_metadata=cohort_metadata,
        )

        template_query = lowercase_placeholder_values(sql_query_result["template_query"])

        return {
            "query_template": template_query,
            "generation_metadata": sql_query_result.get("metadata", {})
        }

    async def generate_query_templates(
            self,
            input_questions: Union[List[str], str],
            drop_first: Optional[bool] = False,
            selected_databases: list[str] = ["OPTUM_CLAIMS_OMOP"],
            cohort_metadata: Optional[CohortMetadata] = None,
            max_concurrency: int = 10  # We can likely handle more concurrency now
    ) -> List[Dict[str, Any]]:
        """
        Generate SQL query templates from input questions with true, efficient parallelism.
        This method now uses the single, pre-initialized instance's resources.
        """
        if isinstance(input_questions, str):
            input_questions = [input_questions]

        # A semaphore is still excellent practice for controlling API calls.
        semaphore = asyncio.Semaphore(max_concurrency)
        log_lock = asyncio.Lock()
        results = [None] * len(input_questions)

        async def process_with_semaphore(idx: int, question: str) -> None:
            """Wrapper to apply the semaphore to our processing method."""
            task_id = f"Task-{idx + 1}"
            # This async with block ensures only `max_concurrency` tasks run at once.
            async with semaphore:
                try:
                    async with log_lock:
                        self.logger.info(
                            f"[{task_id}] STARTED Processing question {idx + 1}/{len(input_questions)}: '{question[:150]}...'")

                    start_time = time.time()

                    template_result = await self._process_single_template_generation(
                        question=question,
                        drop_first=drop_first,
                        selected_databases=selected_databases,
                        cohort_metadata=cohort_metadata
                    )

                    elapsed = time.time() - start_time
                    async with log_lock:
                        self.logger.info(f"[{task_id}] COMPLETED question {idx + 1} in {elapsed:.2f} seconds")

                    results[idx] = {"question": question, **template_result}

                except Exception as e:
                    async with log_lock:
                        self.logger.error(f"[{task_id}] Error for question {idx + 1}: {str(e)}")
                    results[idx] = {"question": question, "query_template": None, "error": str(e)}

        # This part remains the same - it correctly creates and runs the tasks.
        tasks = [
            asyncio.create_task(process_with_semaphore(i, q))
            for i, q in enumerate(input_questions)
        ]

        await asyncio.gather(*tasks)
        return results

    async def post_process_queries(
            self,
            template_results: List[Dict[str, Any]],
            preferred_coding_system: CodingType = CodingType.STANDARD_CODING,
            custom_code_mappings: Optional[Dict[Tuple[str, str], str]] = None,
            max_concurrency: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Post-process query templates in true parallel using isolated workers.
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        log_lock = asyncio.Lock()

        async def process_with_semaphore(template_result: Dict[str, Any]) -> Dict[str, Any]:
            if template_result.get("error") or not template_result.get("query_template"):
                return template_result

            async with semaphore:
                try:
                    # 1. Get a fresh, isolated worker. This is fast and non-blocking.
                    local_rag_agent, local_med_sql_processor = self._create_isolated_worker()

                    # 2. Use the worker to do the job.
                    async with get_db(self.snowflake_database, self.snowflake_database_schema) as db:
                        query_filled = await local_med_sql_processor.post_process_sql_query(
                            template_result["query_template"],
                            preferred_coding_system=preferred_coding_system,
                            rag_agent=local_rag_agent,
                            db_session=db,
                            custom_code_mappings=custom_code_mappings
                        )
                    return {**template_result, "query_filled": query_filled}
                except Exception as e:
                    question = template_result.get('question', 'N/A')
                    async with log_lock:
                        self.logger.error(f"Error post-processing query for question '{question}': {str(e)}")
                    return {**template_result, "query_filled": None, "error": str(e)}

        tasks = [process_with_semaphore(result) for result in template_results]
        processed_results = await asyncio.gather(*tasks)
        return processed_results

    async def execute_queries(
            self,
            processed_results: List[Dict[str, Any]],
            config: Dict[str, Any],
            preferred_coding_system: str = "Standard",
            max_retries: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Execute processed SQL queries and return results in parallel.
        Each query gets its own QA system instance to maintain separate conversation contexts.

        Args:
            processed_results: List of processed query results
            config: Configuration for creating QA system instances
            preferred_coding_system: The coding system used
            max_retries: Maximum number of query retry attempts

        Returns:
            List of dicts with execution results
        """
        async def process_single_execution(processed_result: Dict[str, Any]) -> Dict[str, Any]:
            try:
                qa_system = await self.create_qa_system_for_question(config)
                df, rwd_request = await qa_system.execute_query(
                    processed_result["query_filled"],
                    processed_result["query_template"],
                    processed_result["question"],
                    preferred_coding_system,
                    max_retries
                )
                return {
                    **processed_result,
                    "dataframe": df,
                    "rwd_request": rwd_request
                }
            except Exception as e:
                self.logger.error(f"Error executing query for question '{processed_result['question']}': {str(e)}")
                return {
                    **processed_result,
                    "dataframe": None,
                    "rwd_request": None,
                    "error": str(e)
                }

        tasks = [process_single_execution(processed_result) for processed_result in processed_results]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred during gather
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Exception in query execution for question {i}: {str(result)}")
                final_results.append({
                    **processed_results[i],
                    "dataframe": None,
                    "rwd_request": None,
                    "error": str(result)
                })
            else:
                final_results.append(result)

        return final_results

    async def _process_single_answer_generation(
            self,
            rwd_request: Any
    ) -> str:
        """
        Generates a single answer using an isolated context from the central worker factory.
        """
        # 1. Get a fresh, isolated worker.
        # We only need the agent here, so we use `_` for the processor.
        local_rag_agent, _ = self._create_isolated_worker()

        # 2. Use the isolated agent's 'assistant_answers' to generate the answer.
        await rwd_request.get_answer(local_rag_agent.default_assistant_answers)

        return rwd_request.answer

    async def generate_answers(
            self,
            execution_results: List[Dict[str, Any]],
            max_concurrency: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate answers based on query results in true parallel.
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        log_lock = asyncio.Lock()

        async def process_with_semaphore(execution_result: Dict[str, Any]) -> Dict[str, Any]:
            # If the previous steps failed, just pass the result through.
            if execution_result.get("error") or execution_result.get("rwd_request") is None:
                return execution_result

            async with semaphore:
                try:
                    # This helper method call is correct.
                    answer = await self._process_single_answer_generation(
                        execution_result["rwd_request"]
                    )
                    return {**execution_result, "answer": answer}

                except Exception as e:
                    question = execution_result.get('question', 'N/A')
                    async with log_lock:
                        self.logger.error(f"Error generating answer for question '{question}': {str(e)}")
                    return {**execution_result, "answer": None, "error": str(e)}

        tasks = [process_with_semaphore(result) for result in execution_results]
        final_results = await asyncio.gather(*tasks)

        return final_results


    async def generate_answers_simple(
            self,
            execution_results: List[Dict[str, Any]],
            max_concurrency: int = 10
    ) -> List[str]:
        """
        Generate answers and return only the answer strings.
        """
        full_results = await self.generate_answers(execution_results, max_concurrency)

        answers = []
        for result in full_results:
            if result.get('answer'):
                answers.append(result['answer'])
            else:
                answers.append("")  # Empty string for missing answers

        return answers


    async def answer_questions(
            self,
            input_questions: List[str],
            config: Dict[str, Any],
            drop_first: Optional[bool] = False,
            preferred_coding_system: str = "Standard",
            generate_answer: bool = True,
            custom_code_mappings: Optional[Dict[Tuple[str, str], str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process questions end-to-end and return results in parallel.
        Each question gets its own QA system instance to maintain separate conversation contexts.

        Args:
            input_questions: List of questions to be answered
            config: Configuration for creating QA system instances
            drop_first: True if the first retrieved SQL from the query library should be dropped, False otherwise
                Always use False when deploying, and use True when evaluating/benchmarking the models
            preferred_coding_system: The coding system to use
            generate_answer: Whether to generate natural language answers
            custom_code_mappings: Optional custom code mappings

        Returns:
            List of dicts containing results and metadata for each question
        """
        start_time = time.time()
        
        try:
            # Step 1: Generate query templates
            self.logger.info("Generating query templates...")
            template_results = await self.generate_query_templates(input_questions, drop_first=drop_first)
            
            # Step 2: Post-process queries
            self.logger.info("Post-processing queries...")
            processed_results = await self.post_process_queries(
                template_results,
                preferred_coding_system,
                custom_code_mappings
            )
            
            # Step 3: Execute queries
            self.logger.info("Executing queries...")
            execution_results = await self.execute_queries(
                processed_results,
                config,
                preferred_coding_system
            )
            
            # Step 4: Generate answers if requested
            final_results = execution_results
            if generate_answer:
                self.logger.info("Generating answers...")
                final_results = await self.generate_answers(execution_results)
            
            execution_time = time.time() - start_time
            
            # Add execution time to each result
            for result in final_results:
                result["execution_time"] = execution_time
            
            # Log results
            for result in final_results:
                self.log_results(result)
            
            return final_results

        except Exception as e:
            error_time = time.time() - start_time
            error_results = []
            
            for question in input_questions:
                error_results.append({
                    "question": question,
                    "error": str(e),
                    "execution_time": error_time
                })
            
            self.logger.error(f"Error processing questions: {str(e)}")
            return error_results

    async def get_query_explanations(
            self, 
            sql_queries: List[str], 
            config: Dict[str, Any],
            input_questions: List[str] = None
    ) -> List[str]:
        """
        Get explanations for SQL queries in parallel.
        Each explanation gets its own QA system instance to maintain separate conversation contexts.
        
        Args:
            sql_queries: List of queries
            config: Configuration for creating QA system instances
            input_questions: List of questions (optional)
            
        Returns:
            List of explanations
        """
        async def process_single_explanation(i: int, query: str) -> str:
            try:
                qa_system = await self.create_qa_system_for_question(config)
                question = input_questions[i] if input_questions and i < len(input_questions) else None
                return await qa_system.get_query_explanation(query, question)
            except Exception as e:
                self.logger.error(f"Error getting explanation for query {i}: {str(e)}")
                return f"Error: {str(e)}"
        
        tasks = [process_single_explanation(i, query) for i, query in enumerate(sql_queries)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_explanations = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Exception in explanation generation for query {i}: {str(result)}")
                final_explanations.append(f"Error: {str(result)}")
            else:
                final_explanations.append(result)
        
        return final_explanations

    def log_results(self, result: dict) -> None:
        """Log the results of the question answering process"""
        self.logger.info(f"Question: {result.get('question', 'N/A')}")
        self.logger.info(f"SQL template: {result.get('query_template', 'N/A')}")
        self.logger.info(f"SQL filled: {result.get('query_filled', 'N/A')}")
        self.logger.info(f"Answer: {result.get('answer', 'N/A')}")
        self.logger.info(f"Total time: {result.get('execution_time', 0):.1f} s")
        if result.get('error'):
            self.logger.error(f"Error: {result['error']}")
        self.logger.info("-----------------")
