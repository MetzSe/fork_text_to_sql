import asyncio
import functools
import json
import logging
import os
import re
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from snowflake.connector.cursor import SnowflakeCursor
import warnings

import ascent_ai.models.inference.prompts as prompts
from ascent_ai.data.queries import querylib_loader
from ascent_ai.models.inference.assistants import create_assistant
from ascent_ai.models.inference.rag import RAGConfig, RAGProcessor
from ascent_ai.models.inference.rwd_request import RWDRequest
from ascent_ai.data.processing.sql_post_processor import MedicalSQLProcessor
from ascent_ai.db.snowflake_session import get_db
from ascent_ai.external_calls import MedicalCoder
from ascent_ai.config.settings import settings
from ascent_ai.models.inference.sql_verify import verify_and_correct_query
from ascent_ai.utils.mixing_temp import (
    add_messages_to_assistant,
    get_text_sql_template_for_rag,
)
from ascent_ai.data.processing.sql_post_processor import NoSQLFoundError
from ascent_ai.utils.mixing_temp import prepare_rwd_request
from ascent_ai.models.inference.sql_explainer import SQLExplainer
from ascent_ai.data.processing.utils_processing import lowercase_placeholder_values

logger = logging.getLogger(__name__)


class CriteriaProcessor:
    VALID_ASSISTANT_TYPES = {
        "claude_sonnet",
        "gpt",
        "gpt-o1",
        "gemini",
        "deepseek",
        "bedrock_llama3",
    }

    def __init__(
        self,
        text2sql_assistant_type: str = "claude_sonnet",
        mask_assistant_type: str = "claude_sonnet",
        text_to_criteria_assistant_type: str = "claude_sonnet",
        text2sql_assistant_model_name: str = None,
        mask_assistant_model_name: str = None,
        text_to_criteria_assistant_model_name: str = None,
    ):
        self.text_to_criteria_assistant_type = text_to_criteria_assistant_type
        self.mask_assistant_type = mask_assistant_type
        self.text2sql_assistant_type = text2sql_assistant_type

        # Store model names
        self.text_to_criteria_assistant_model_name = (
            text_to_criteria_assistant_model_name
        )
        self.mask_assistant_model_name = mask_assistant_model_name
        self.text2sql_assistant_model_name = text2sql_assistant_model_name

        # Initialize assistants
        self.mask_assistant = self._create_validated_assistant(
            mask_assistant_type,
            "mask_assistant_type",
            model_name=mask_assistant_model_name,
        )
        self.text_to_criteria_assistant = self._create_validated_assistant(
            text_to_criteria_assistant_type,
            "text_to_criteria_assistant_type",
            model_name=text_to_criteria_assistant_model_name,
        )
        self.text2sql_assistant = self._create_validated_assistant(
            text2sql_assistant_type,
            "text2sql_assistant_type",
            model_name=text2sql_assistant_model_name,
        )

        # Medical coder is not implemented - set recommender to None
        recommender = None
        logger.info("MedicalCoder not implemented - using None for recommender")

        self.med_sql_processor = MedicalSQLProcessor(
            assistant=self.text2sql_assistant, recommender=recommender
        )
        self.sql_explainer = SQLExplainer()

        main_path = os.getcwd()
        querylib_file = querylib_loader.get_latest_querylib_file(main_path)
        log_folder = os.path.join(os.path.dirname(main_path), "scripts", "logs")
        prompt_gpt_cleaned = self.remove_question_and_below(prompts.prompt_gpt)
        prompt_template = prompt_gpt_cleaned + prompts.criteria_to_sql_prompt_add_on
        config = RAGConfig(
            main_path=main_path,
            log_folder=log_folder,
            querylib_file=querylib_file,
            prompt_template=prompt_template,
            assistant_type=self.text2sql_assistant_type,
            model_name=self.text2sql_assistant_model_name,
            top_k_prompt=5,
        )
        self.rag_agent = RAGProcessor(config)

    def _create_validated_assistant(
        self, assistant_type: str, param_name: str, model_name: str = None
    ):
        validated_type = self._validate_assistant_type(assistant_type, param_name)
        return create_assistant(assistant_type=validated_type, model_name=model_name)

    @classmethod
    def _validate_assistant_type(cls, assistant_type: str, param_name: str) -> str:
        if assistant_type not in cls.VALID_ASSISTANT_TYPES:
            warnings.warn(
                f"Invalid {param_name}: {assistant_type}. "
                f"Defaulting to 'claude_sonnet'. "
                f"Valid types are: {cls.VALID_ASSISTANT_TYPES}",
                UserWarning,
            )
            return "claude_sonnet"
        return assistant_type

    @staticmethod
    def clean_sql_blocks(json_str):
        """
        Clean and format JSON strings containing SQL code blocks.
        Converts ```sql blocks into proper JSON-formatted strings.
        """

        def replace_sql_block(match):
            """Format SQL block as a proper JSON key-value pair"""
            key = match.group(1)
            sql_content = match.group(2).strip()
            escaped_sql = sql_content.replace("\n", "\\n").replace('"', '\\"')
            return f'"{key}": "{escaped_sql}"'

        # Match pattern: "key": ```sql [content] ```
        sql_block_pattern = r'"([^"]+)":\s*```sql\s*(.*?)\s*```\s*(?=,|\})'

        # Step 1: Replace SQL blocks with properly formatted JSON
        cleaned_json = re.sub(
            sql_block_pattern, replace_sql_block, json_str, flags=re.DOTALL
        )

        # Step 2: Clean up formatting artifacts
        cleaned_json = cleaned_json.replace("```json", "").replace("```", "").strip()

        # Step 3: Normalize whitespace
        cleaned_json = re.sub(r"\s+", " ", cleaned_json)

        # Step 4: Ensure proper JSON structure with braces
        if not cleaned_json.startswith("{"):
            cleaned_json = "{" + cleaned_json
        if not cleaned_json.endswith("}"):
            cleaned_json = cleaned_json + "}"

        return cleaned_json

    async def split_sql_into_queries(self, complete_sql_query, incl_excl_criteria):
        # only extract the part of the prompts that contained the generation instructions
        instruction_prompt = prompts.prompt_gpt.split(
            "# This is the question you need to provide the SQL for:"
        )[0]
        criteria_prompt = prompts.criteria_to_sql_prompt_add_on.split(
            "Each query must return two columns: "
        )[0]

        # sql_splitting_prompt_filled = prompts.sql_splitting_prompt.format(
        #     complete_sql_query=complete_sql_query, incl_excl_criteria=incl_excl_criteria
        # )

        sql_splitting_prompt_filled = prompts.sql_splitting_prompt.replace(
            "${incl_excl_criteria}", incl_excl_criteria
        ).replace(
            "${complete_sql_query}", complete_sql_query
        )

        prompt_filled = (
            f""" Please find below the instructions used to generate the query you need to split.
        Please respect the conventions and rules in your split queries."""
            + "\n"
            + instruction_prompt
            + "\n"
            + criteria_prompt
            + "\n Instruction on splitting task: \n"
            + sql_splitting_prompt_filled
        )

        split_sql_json = await self.text2sql_assistant.get_response(prompt_filled)
        sql_split_cleaned_2 = self.clean_sql_blocks(split_sql_json)
        sql_split_cleaned_3 = re.sub(
            r"\bLIMIT\s+\d+\b", "", sql_split_cleaned_2, flags=re.IGNORECASE
        )

        sql_split_cleaned = self.fix_json_linebreaks(sql_split_cleaned_3)

        try:
            sql_split_dict = json.loads(sql_split_cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            logger.error(f"Attempted to parse: {sql_split_cleaned}")
            raise

        query_template_preds = list(sql_split_dict.values())

        return query_template_preds

    @staticmethod
    def get_patient_funnel_from_dfs(output_dfs, criteria_list):
        # Initialize the final set of person_ids with the first inclusion criteria
        final_set = (
            set(output_dfs[0]["person_id"]) if not output_dfs[0].empty else set()
        )

        # List to keep track of the number of patients after each step
        patient_counts = [len(final_set)]

        # Apply each criterion in the specified order
        for criterion, df in zip(criteria_list[1:], output_dfs[1:]):
            if df.empty:
                # If the DataFrame is empty, no change to final_set
                pass
            elif criterion == "include":
                final_set &= set(df["person_id"])
            elif criterion == "exclude":
                final_set -= set(df["person_id"])
            patient_counts.append(len(final_set))

        logger.info(f"Patient counts: {patient_counts}")
        return patient_counts, final_set

    @staticmethod
    def plot_funnel_with_criteria(
        patient_counts,
        criteria_type_list,
        criteria_text_list,
        char_limit=200,
        output_folder=None,
        streamlit_mode=False,
    ):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the funnel plot
        steps = ["Initial Inclusion"] + [
            f"Step {i + 1} ({criterion})"
            for i, criterion in enumerate(criteria_type_list[1:])
        ]

        ax.plot(range(len(steps)), patient_counts, marker="o")

        # Add value labels above each point
        for i, count in enumerate(patient_counts):
            ax.annotate(
                f"{count}",
                (i, count),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        ax.set_title("Funnel Plot")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Number of Patients")
        ax.grid(True)

        # Set tick positions explicitly
        ax.set_xticks(range(len(steps)))

        # Set tick labels and rotate them
        ax.set_xticklabels(steps, rotation=15, ha="right")

        criteria_text = "\n".join(
            [
                f"Criteria {i}: {criterion[:char_limit] + '...' if len(criterion) > char_limit else criterion}"
                for i, criterion in enumerate(criteria_text_list)
            ]
        )

        # Adjust the bottom margin to make room for the criteria text
        plt.subplots_adjust(bottom=0.4)

        # Add the criteria text below the plot
        fig.text(
            0.5,
            0.01,
            criteria_text,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="center",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Rotate x-axis labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")

        # Adjust layout to prevent cutting off labels
        # plt.tight_layout()

        if output_folder:
            plt.savefig(f"{output_folder}/funnel_plot.png")

        if streamlit_mode:
            return fig
        else:
            plt.show()

    def find_medical_concepts(self, query_template: str, database: str):
        """Find medical concepts mentioned in the given query template"""
        placeholder_types = [
            "condition",
            "procedure",
            "measurement",
            "drug",
            "drug_class",
            "observation",
            "visit",
        ]
        placeholder_pattern = r"\[(" + "|".join(placeholder_types) + r")@([^\]]+)\]"
        placeholders = {
            (m.group(1).lower(), m.group(2).lower())
            for m in re.finditer(
                placeholder_pattern, query_template, flags=re.IGNORECASE
            )
        }
        concepts = self.med_sql_processor.process_and_restructure(database)
        concepts_filtered = [
            c
            for c in concepts
            if (c["category"].lower(), c["name"].lower()) in placeholders
        ]
        return concepts_filtered

    @staticmethod
    def fix_json_linebreaks(json_string):
        # This pattern matches the SQL query part of each JSON key-value pair
        pattern = r':\s*"(.*?)"(?=,|\s*})'

        def replace_newlines(match):
            # Replace newlines with '\n' in the SQL query
            return (
                ': "' + re.sub(r"\s+", " ", match.group(1).replace("\n", "\\n")) + '"'
            )

        # Apply the replacement
        fixed_json_string = re.sub(
            pattern, replace_newlines, json_string, flags=re.DOTALL
        )
        return fixed_json_string

    async def _run_sub_query(
        self,
        database: str,
        query_template_pred: str,
        preferred_coding_system: str,
        schema_name: Optional[str] = None,
        custom_code_mappings: Optional[Dict[Tuple[str, str], str]] = None,
    ):
        async with get_db(database, schema_name) as db:

            query_filled_pred = await self.med_sql_processor.post_process_sql_query(
                query_template_pred,
                preferred_coding_system=preferred_coding_system,
                rag_agent=self.rag_agent,
                db_session=db,
                custom_code_mappings=custom_code_mappings,
            )

            rwd_request_pred = RWDRequest(
                user_input="",
                query_filled=query_filled_pred,
                query_template=query_template_pred,
            )
            rwd_request_pred.med_sql_processor = self.med_sql_processor
            rwd_request_pred.rag = self.rag_agent

            logger.debug(f"Query template pred: {query_template_pred}")
            logger.debug(f"Query filled pred: {query_filled_pred}")

            df = await rwd_request_pred.run_query(
                db=db,
                max_retries=5,
                reset_conversation=False,
            )

        if df is not None:
            if "person_id" in df.columns and not df.empty:
                logger.debug(f"Unique patients in cohort: {df['person_id'].nunique()}")

        return query_filled_pred, df

    async def split_query_by_criteria(
        self,
        preferred_coding_system,
        snowflake_database,
        query_template_pred,
        criteria_dict,
        custom_code_mappings=None,
        snowflake_database_schema=None,
    ):
        # Filter criteria_dict to only include 'include' and 'exclude' keys
        filtered_criteria = {
            k: v for k, v in criteria_dict.items() if k in ["include", "exclude"]
        }

        criteria_type_list = [
            key for key, values in filtered_criteria.items() for _ in values
        ]
        criteria_text_list = [
            item for sublist in filtered_criteria.values() for item in sublist
        ]

        logger.info("Starting to split the query into multiple template subqueries")
        query_template_preds = await self.split_sql_into_queries(
            query_template_pred, str(criteria_dict)
        )

        # Since we're working with templates only, set filled queries to None to avoid confusion
        query_filled_preds = None
        logger.info("Template-only mode: query_filled_preds set to None (medical coder bypass)")

        # Return None for output_dfs since we're not executing queries
        output_dfs = [None] * len(query_template_preds)
        logger.info(f"Generated {len(query_template_preds)} template queries for funnel analysis")

        return (
            query_template_preds,
            query_filled_preds,
            output_dfs,
            criteria_type_list,
            criteria_text_list,
        )


    @staticmethod
    async def split_text_into_criteria(assistant, input_text):
        prompt = prompts.claim_extraction_prompt + input_text
        split_text_json = await assistant.get_response(prompt)
        return split_text_json

    async def text_to_criteria(self, input_text):
        # split criteria and loads them as json
        incl_excl_criteria = await self.split_text_into_criteria(
            assistant=self.text_to_criteria_assistant, input_text=input_text
        )
        # only keep text between curly brackets
        # Extract only the text between curly brackets
        pattern = r"\{[\s\S]*\}"
        match = re.search(pattern, incl_excl_criteria)

        if match:
            incl_excl_criteria = match.group()
        else:
            raise ValueError("No valid JSON structure found in the input.")

        # Remove any remaining JSON code block markers if present
        incl_excl_criteria = incl_excl_criteria.replace("```json\n", "").replace(
            "\n```", ""
        )

        incl_excl_criteria = self.fix_json_linebreaks(incl_excl_criteria)

        try:
            criteria_dict = json.loads(incl_excl_criteria)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON structure: {e}")

        return criteria_dict

    @staticmethod
    def remove_question_and_below(text):
        pattern = r"#\s*Question.*$"
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)

        if match:
            return text[: match.start()].strip()
        else:
            print("Warning: No match found for '# Question' or similar pattern.")
            return text.strip()

    async def criteria_to_data(
        self,
        criteria_dict,
        preferred_coding_system,
        snowflake_database,
        main_path,
        log_folder,
        querylib_file,
        rag_assistant_type="claude_sonnet",
        rag_model_name="us.anthropic.claude-sonnet-4-20250514-v1:0",
        custom_code_mappings: Optional[Dict[Tuple[str, str], str]] = None,
        top_k_prompt=5,
        max_nb_row=None,
        run_query=True,
        verify=False,
        use_rag=True,
        use_rag_queries=True,
        use_rag_cohort=True,
        drop_first=False,
        rag_random=False,
        verify_agent_type="claude_sonnet",
        verify_agent_model_name="us.anthropic.claude-sonnet-4-20250514-v1:0",
        max_attempt_verify=3,
        snowflake_database_schema=None,
    ):

        verification_results = None
        verdict = None

        prompt_gpt_cleaned = self.remove_question_and_below(prompts.prompt_gpt)
        prompt_template = prompt_gpt_cleaned + prompts.criteria_to_sql_prompt_add_on
        config = RAGConfig(
            main_path=main_path,
            log_folder=log_folder,
            querylib_file=querylib_file,
            prompt_template=prompt_template,
            assistant_type=rag_assistant_type,
            model_name=rag_model_name,
            top_k_prompt=top_k_prompt,
        )
        rag_agent = RAGProcessor(config)

        logger.info(
            f"RAG assistant type: {rag_assistant_type}" f"model name: {rag_model_name}"
        )

        query_template_pred, prompt, df_recs_list_out, question_masked = (
            await self.prepare_query_template(
                rag_agent,
                criteria_dict,
                use_rag=use_rag,
                use_rag_cohort=use_rag_cohort,
                use_rag_queries=use_rag_queries,
                drop_first=drop_first,
                rag_random=rag_random,
            )
        )
        query_template_pred = lowercase_placeholder_values(query_template_pred)

        original_prompt = prompt

        med_sql_processor = MedicalSQLProcessor(assistant=None, recommender=None)
        if verify:
            logger.info(f"Performing verification for cohort generator...")
            verify_agent = create_assistant(
                assistant_type=verify_agent_type, model_name=verify_agent_model_name
            )
            logger.info(
                f"Verifying agent: {verify_agent_type} {verify_agent_model_name}"
            )

            def format_value(value):
                if isinstance(value, (list, dict, set, tuple)):
                    return str(value)
                elif value is None:
                    return "None"
                else:
                    return str(value)

            user_input = "\n".join(
                f"{key} {format_value(value)}" for key, value in criteria_dict.items()
            )

            (
                query_template_pred,
                user_input,
                verdict,
                verification_results,
            ) = await verify_and_correct_query(
                query_template_pred=query_template_pred,
                user_input=user_input,
                verify_agent=verify_agent,
                med_sql_processor=med_sql_processor,
                max_attempt_verify=max_attempt_verify,
                original_prompt=original_prompt,
            )
            logger.info(f"Query verification result: {verdict}")

        # Since medical coder is mockup, use template directly as "filled" query
        query_filled = query_template_pred
        logger.info("Using template as filled query (medical coder bypass)")

        if run_query:
            message = f"Template query generated successfully. Query execution skipped (template-only mode)."
            df_patients = None  # No patient data since we're not executing queries
            sql_executed = False
            logger.info("Template-only mode: Query execution skipped")
        else:
            message, df_patients, sql_executed = None, None, False

        return (
            message,
            query_template_pred,
            query_filled,
            df_patients,
            df_recs_list_out,
            question_masked,
            rag_agent,
            verification_results,
            verdict,
            sql_executed,
        )

    async def prepare_query_template(
        self,
        rag_agent: RAGProcessor,
        criteria_dict,
        use_rag=True,
        use_rag_queries=True,
        use_rag_cohort=True,
        drop_first=False,
        rag_random=False,
    ) -> tuple[Any, Any, Any, Any]:
        input_question = str(criteria_dict)
        logger.info(f"Input inclusion/exclusion criteria: {input_question}")

        if use_rag:
            initial_prompt, text_sql_template, df_recs_list_out, question_masked = (
                await self.process_criteria_to_counts(
                    input_question,
                    rag_agent,
                    mask_assistant=self.mask_assistant,
                    use_rag_queries=use_rag_queries,
                    use_rag_cohort=use_rag_cohort,
                    drop_first=drop_first,
                    rag_random=rag_random,
                )
            )
            rag_prompt = initial_prompt + text_sql_template
        else:
            initial_prompt = rag_agent.config.prompt_template.replace(
                "${incl_excl_criteria}", str(criteria_dict)
            )
            rag_prompt = initial_prompt
            df_recs_list_out = None
            question_masked = None

        rag_assistant_answer = await rag_agent.default_assistant.get_response(
            rag_prompt
        )

        try:
            query_template_pred = self.med_sql_processor.parse_sql_from_response(
                rag_assistant_answer
            )
            return query_template_pred, rag_prompt, df_recs_list_out, question_masked

        except NoSQLFoundError as e:
            logger.error(f"Failed to parse SQL: {e}")
            logger.error(f"Assistant answer: {rag_assistant_answer}")
            raise

    async def query_cohort(
        self,
        db: SnowflakeCursor,
        criteria_dict,
        rag_agent: RAGProcessor,
        query: str,
        query_template: str,
        max_nb_row: int | None = None,
    ):
        input_question = str(criteria_dict)
        logger.info(f"Input question: {input_question}")

        logger.info("Complete filled query generated")
        logger.info(query)

        logger.info("Trying to run resulting query...")
        rwd_request_pred = prepare_rwd_request(
            user_input=input_question,
            query_filled_pred=query,
            query_template_pred=query_template,
            question_masked="",
            rag=rag_agent,
            df_recs_list_out=pd.DataFrame(columns=["Score"]),
            new_prompt=rag_agent.default_assistant.conversation,
            med_sql_processor=self.med_sql_processor,
        )

        patients = await rwd_request_pred.run_query(
            db=db,
            max_retries=5,
            reset_conversation=False,
        )
        logger.info("Data retrieved.")

        if max_nb_row is not None and patients.shape[0] > max_nb_row:
            logger.info(
                f"Truncating cohort to {max_nb_row} records (original size: {patients.shape[0]})"
            )
            patients = patients[:max_nb_row]
        else:
            logger.info(f"Cohort size: {patients.shape[0]} records")

        if (
            "index_date" in patients.columns
            and (null_records := patients["index_date"].isnull().sum()) > 0
        ):
            message = f"Cohort contains {null_records} ({null_records / patients.shape[0]:.2%}) null records in index date column!"
            logger.warning(message)
        else:
            message = None

        logger.info("Done")

        return message, patients, rwd_request_pred.sql_executed

    @staticmethod
    async def create_distribution_tasks_for_cohorts(table_one, database_name, schema_name=None):
        async def run_task(func, label, max_retries=3, retry_delay=1):
            attempt = 0
            while attempt < max_retries:
                try:
                    async with get_db(database_name, schema_name) as db:
                        result = await func(db, database_name)
                        return {label: json.loads(result)}
                except Exception as e:
                    attempt += 1
                    logger.warning(
                        f"An error occurred, retrying ({attempt}/{max_retries})... Error: {e}"
                    )
                    if attempt >= max_retries:
                        logger.exception(
                            f"Max retries reached. An error occurred during task execution: {e}"
                        )
                        raise e
                    await asyncio.sleep(retry_delay)

        tasks = [
            run_task(table_one.get_gender_distribution, label="gender_distribution"),
            run_task(
                table_one.get_ethnicity_distribution, label="ethnicity_distribution"
            ),
            run_task(table_one.get_race_distribution, label="race_distribution"),
            run_task(
                table_one.get_top_k_diseases_distribution,
                label="top_diseases_distribution",
            ),
            run_task(
                table_one.get_top_k_drugs_distribution, label="top_drugs_distribution"
            ),
            run_task(
                table_one.get_top_k_drug_eras_by_ingredient_distribution,
                label="top_drug_eras_distribution",
            ),
        ]

        return tasks

    @staticmethod
    async def process_criteria_to_counts(
        criteria_dict: Union[Dict, str],
        rag_agent: RAGProcessor,
        mask_assistant=None,
        use_rag_queries=True,
        use_rag_cohort=True,
        drop_first=False,
        rag_random=False,
    ) -> Tuple:
        """Process criteria dictionary to get counts"""
        if mask_assistant is None:
            logger.warning(
                "mask assistant not defined. Using RAG agent assistant instead."
            )
            mask_assistant = rag_agent.default_assistant

        question_masked, question = await rag_agent.querylib.get_masked_question(
            question=str(criteria_dict), assistant=mask_assistant
        )

        initial_prompt = rag_agent.config.prompt_template.replace(
            "${incl_excl_criteria}", str(criteria_dict)
        )

        text_sql_template_all = ""
        df_recs_list_out_cohort_gen = pd.DataFrame()
        df_recs_list_out_all = pd.DataFrame()

        if use_rag_cohort:
            # first retrieve the examples queries for the cohort generator
            text_sql_template_cohort_gen, df_recs_list_out_cohort_gen = (
                await get_text_sql_template_for_rag(
                    question_masked=str(criteria_dict),
                    rag=rag_agent,
                    question_type="COHORT_GENERATOR",
                    drop_first=drop_first,
                    rag_random=rag_random,
                )
            )

            text_sql_template_all += text_sql_template_cohort_gen + "\n"

        if use_rag_queries:
            criteria_dict_masked = CriteriaParser.parse_string_to_dict(question_masked)
            criteria_list = CriteriaParser.extract_criteria(criteria_dict_masked)

            logger.info("Retrieve relevant queries for each criteria")
            df_recs_list_out_queries = []
            for criteria in criteria_list:
                text_sql_template, df_recs_list_out = (
                    await get_text_sql_template_for_rag(
                        question_masked=str(criteria), rag=rag_agent, question_type="QA"
                    )
                )

                text_sql_template_all += text_sql_template + "\n"
                df_recs_list_out_queries.append(df_recs_list_out)

            # Concatenate all queries corresponding to the criteria dataframes
            if df_recs_list_out_queries:
                df_recs_list_out_all = pd.concat(
                    df_recs_list_out_queries, ignore_index=True
                )

        # Handle different combinations of flags
        if use_rag_queries and use_rag_cohort:
            final_df = pd.concat(
                [df_recs_list_out_all, df_recs_list_out_cohort_gen], ignore_index=True
            )
        elif use_rag_queries:
            final_df = df_recs_list_out_all
        elif use_rag_cohort:
            final_df = df_recs_list_out_cohort_gen
        else:
            logger.warning(
                "Both use_rag_queries and use_rag_cohort are False. Returning empty DataFrame."
            )
            final_df = pd.DataFrame()

        # Remove duplicates if there are any records
        if not final_df.empty:
            rows_before = len(final_df)
            final_df = final_df.drop_duplicates(keep="last")
            rows_dropped = rows_before - len(final_df)

            if rows_dropped > 0:
                logger.info(
                    f"Dropped {rows_dropped} duplicate samples from query library"
                )

        add_messages_to_assistant(
            [initial_prompt, text_sql_template_all], rag_agent.default_assistant
        )

        return initial_prompt, text_sql_template_all, final_df, question_masked

    async def get_query_explanation(
        self, sql_query: str, criteria_dict: dict = None
    ) -> str:
        return await self.sql_explainer.get_explanation(sql_query, str(criteria_dict))


def handle_streamlit_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON in criteria: {str(e)}")
        except requests.HTTPError as e:
            st.error(f"HTTP Error: {e.response.status_code}")
            try:
                error_detail = e.response.json()
                if isinstance(error_detail.get("detail"), dict):
                    st.error(
                        f"Error: {error_detail['detail'].get('error', 'Unknown error')}"
                    )
                    st.error("Traceback:")
                    st.code(
                        error_detail["detail"].get(
                            "traceback", "No traceback available"
                        )
                    )
                else:
                    st.error(
                        f"Error detail: {error_detail.get('detail', 'No detail provided')}"
                    )
            except json.JSONDecodeError:
                st.error(f"Response content (non-JSON): {e.response.text}")
        except Exception as e:
            error_detail = {"error": str(e), "traceback": traceback.format_exc()}
            st.error(f"An unexpected error occurred: {error_detail['error']}")
            st.code(error_detail["traceback"])

    return wrapper


class CriteriaParser:
    @staticmethod
    def extract_criteria(criteria_dict: Union[Dict, str]) -> List:
        """Extract criteria from dictionary or string"""
        if isinstance(criteria_dict, str):
            criteria_dict = CriteriaParser.parse_string_to_dict(criteria_dict)

        try:
            return criteria_dict["include"] + criteria_dict["exclude"]
        except KeyError:
            return (
                criteria_dict["masked_text"]["include"]
                + criteria_dict["masked_text"]["exclude"]
            )

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


@handle_streamlit_errors
def convert_criteria_to_data(criteria_text, hardcode_concepts, selected_dict, base_url):
    with st.spinner("Parsing input criteria..."):
        payload = {"input_text": criteria_text, "assistant_type": "gpt"}

        response = requests.post(f"{base_url}/text-to-criteria", json=payload)
        response.raise_for_status()
        criteria_dict = response.json()

        if "include" in criteria_dict:
            with st.expander("Inclusion Criteria", expanded=True):
                st.json(criteria_dict["include"])

        if "exclude" in criteria_dict:
            with st.expander("Exclusion Criteria", expanded=True):
                st.json(criteria_dict["exclude"])

        st.success("Input criteria parsed")

    with st.spinner("Converting criteria to query and retrieve data..."):
        payload = {
            "criteria_dict": criteria_dict,
            "hardcode_concepts": hardcode_concepts,
        }

        if hardcode_concepts and selected_dict:
            payload["substitution_dict"] = selected_dict

        response = requests.post(f"{base_url}/criteria2data", json=payload)
        response.raise_for_status()
        data = response.json()

        if "df_patients" in data:
            with st.expander("Data retrieved", expanded=True):
                df_patients = pd.DataFrame(data["df_patients"])
                st.dataframe(df_patients)

        other_data = {
            k: v
            for k, v in data.items()
            if k not in ["df_patients", "incl_excl_criteria"]
        }
        if other_data:
            with st.expander("Other Information", expanded=False):
                st.json(other_data)

        st.success("Data retrieved")

    return data


@handle_streamlit_errors
def get_patient_funnel(
    query_with_placeholders, criteria_dict, hardcode_concepts, selected_dict, base_url
):
    with st.spinner("Analyzing query and creating funnel..."):
        payload = {
            "query_with_placeholders": query_with_placeholders,
            "criteria_dict": criteria_dict,
            "hardcode_concepts": hardcode_concepts,
        }

        if hardcode_concepts and selected_dict:
            payload["substitution_dict"] = selected_dict

        response = requests.post(f"{base_url}/split-criteria-sql", json=payload)
        response.raise_for_status()
        data = response.json()

        if "criteria_text_list" in data:
            with st.expander("Criteria Text List", expanded=True):
                st.json(data["criteria_text_list"])
        if "criteria_type_list" in data:
            with st.expander("Criteria Type List", expanded=True):
                st.json(data["criteria_type_list"])

        if "patient_counts" in data:
            with st.expander("Patient counts", expanded=True):
                st.json(data["patient_counts"])

        other_data = {
            k: v
            for k, v in data.items()
            if k not in ["criteria_text_list", "criteria_type_list", "patient_counts"]
        }
        if other_data:
            with st.expander("Other Information", expanded=False):
                st.json(other_data)

        st.success("Patient funnel generated.")

    return data
