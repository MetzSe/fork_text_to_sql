# coding=utf-8
__author__ = "Angelo Ziletti"
__maintainer__ = "Angelo Ziletti"
__email__ = "angelo.ziletti@bayer.com"
__date__ = "22/02/24"

import logging
from typing import Optional
import pandas as pd


from ascent_ai.models.inference.prompts import cohort_creation_prompt
from ascent_ai.models.inference.assistants import create_assistant


logger = logging.getLogger(__name__)


class Table1:
    def __init__(self, input_question: Optional[str] = None):
        self.input_question = input_question
        self.input_text_cohort_generator = None

        self.cohort_creation_question = None
        self.cohort_df = {}

        # derived quantities
        self.rwd_request = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the non-pickleable 'assistant' attribute from the state
        if "assistant" in state:
            del state["assistant"]
        return state

    def add_cohort(self, input_cohort_df, database):
        # Check for the presence of 'patient_id' column
        if "patient_id" in input_cohort_df.columns:
            # Rename 'patient_id' column to 'person_id'
            logger.info("Renaming 'patient_id' column to 'person_id'.")
            input_cohort_df.rename(columns={"patient_id": "person_id"}, inplace=True)
        elif "person_id" not in input_cohort_df.columns:
            logger.error("Expected 'patient_id' or 'person_id' column not found. Please check the input DataFrame.")
            raise

        # Check if 'index_date' column exists
        if "index_date" in input_cohort_df.columns:
            # Convert 'index_date' to datetime
            try:
                input_cohort_df["index_date"] = pd.to_datetime(input_cohort_df["index_date"])
            except Exception as e:
                logger.warning(f"Error converting 'index_date' to datetime: {e}. 'index_date' will remain as is.")
        else:
            logger.error("'index_date' column not found in the input DataFrame. Proceeding without it.")
            raise

        # Store the modified DataFrame in the cohort_df
        self.cohort_df[database] = input_cohort_df

    async def get_cohort_creation_question(self, input_question):
        prompt_filled = cohort_creation_prompt.replace("${question}", input_question.replace("'", "\\'"))
        assistant_type = "claude_sonnet"
        model_name = "us.anthropic.claude-sonnet-4-20250514-v1:0"

        input_criteria_assistant = create_assistant(assistant_type=assistant_type, model_name=model_name)

        input_text_cohort_generator = await input_criteria_assistant.get_response(prompt_filled)

        self.input_text_cohort_generator = input_text_cohort_generator

        return input_text_cohort_generator
