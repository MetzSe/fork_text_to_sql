# coding=utf-8
__author__ = "Angelo Ziletti"
__maintainer__ = "Angelo Ziletti"
__date__ = "26/05/25"

import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import logging
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any, Union, Dict
import time


from ascent_ai.data.queries.query_library import QueryLibrary
from ascent_ai.models.evaluation.eval import get_subset_df
from ascent_ai.models.inference.assistants import create_assistant
from ascent_ai.models.inference.rag import QueryLibraryManager

logger = logging.getLogger(__name__)


class QuestionFilterResponse(BaseModel):
    """Schema for the question filter response"""
    rationale: str = Field(..., description="Explanation of the decision")
    answerable: bool = Field(..., description="Whether the question can be answered based on the query library")
    answerable_score: float = Field(
        ...,
        description="Score between 0.0 (definitely not answerable) and 1.0 (definitely answerable)",
        ge=0.0,
        le=1.0,
    )
    suggested_questions: List[str] = Field(..., description="List of suggested alternative questions "
                                                            "if not answerable else Empty")

    @field_validator("answerable_score")
    def validate_answerable_score(cls, v):
        if v < 0.0 or v > 1.0:
            logger.warning(
                f"Confidence score {v} is outside valid range [0,1]. Clamping to valid range."
            )
            return max(0.0, min(1.0, v))
        return v


async def filter_question(
        question: str,
        query_library_questions: List[str],
        assistant: Optional[Any] = None,
        assistant_type: str = "gemini",
        model_name: str = "gemini-2.5-flash",
) -> QuestionFilterResponse:
    """
    Filter a question to determine if it can be answered based on the query library.

    Args:
        question: The question to filter
        query_library_questions: List of questions from the query library for context
        assistant: LLM assistant (if None, one will be created)
        assistant_type: Type of assistant to create if none provided
        model_name: Model name to use when creating assistant

    Returns:
        QuestionFilterResponse: Structured response with rationale, answerable flag, confidence score, and suggested questions
    """
    # Create assistant if not provided
    if assistant is None:
        assistant = create_assistant(
            assistant_type=assistant_type, model_name=model_name
        )

    question_filter_prompt = """
    We have a system that answers epidemiological questions based on data by generating text-to-SQL translation 
    via a large language model (LLM) and retrieval augmented generation (RAG).
    Based on our question-SQL library, these are the questions that we can answer with certainty: {query_library_questions}
    The question-SQL library gives a good snapshots of our capabilities.
    The goal is to filter out questions that cannot be answered based on our question-SQL library.

    Our system can handle visualization and plotting requests even if they don't appear in the question-SQL library examples. 
    Any question that asks for plots, charts, graphs, or visualizations of data that can be retrieved should be 
    considered answerable.

    The system is not able to provide answers to question which include geographic location or countries.
    e.g. "How many people have CONDITION in the US?"
    If the geographic location matches one of the above, we should return in the rationale field to select one of the 
    relevant databases from the Ascent app, and in the suggested questions the same question but without the location.

    "United Kingdom": ["CPRD_EHR_AURUM", "CPRD_EHR_AURUM_OMOP", "CPRD_EHR_GOLD", "CPRD_EHR_GOLD_OMOP"],
    "Germany": ["FDZ_CLAIMS_PUBLIC_USE_FILE"],
    "USA": ["FLATIRON_EHR_LC", "ICPSR_EHR_SWAN", "KOMODO_CLAIMS", "MKTSCAN_CLAIMS", "MKTSCAN_CLAIMS_OMOP",
            "OPTUM_CLAIMS", "OPTUM_CLAIMS_OMOP", "OPTUM_EHR_5PCT_OMOP", "OPTUM_EHR_CKD", "OPTUM_EHR_CKD_OMOP",
            "OPTUM_EHR_HF", "OPTUM_EHR_HF_OMOP", "OPTUM_EHR_SAMPLE", "STATINMED_CLAIMS_PC", "STATINMED_CLAIMS_PD",
            "STATINMED_CLAIMS_SAMPLE", "VERANTOS_EHR_AIS_OMOP"],
    "Japan": ["MDV_CLAIMS_DM", "MDV_CLAIMS_DM_OMOP", "MDV_CLAIMS_HF", "MDV_CLAIMS_HF_OMOP", "RWDCO_CLAIMS_EHR_CKD_OMOP",
            "RWDCO_CLAIMS_EHR_HF_OMOP"]

    In your evaluation, please also take into account the generalization ability of current LLMs:
    1. Questions that are similar from the ones in the RAG library could be answered correctly
    2. Questions asking for visualizations of data that can be retrieved should be considered answerable

    In doubt, consider the input question answerable.

    This is the input question: {input_question}

    Analyze whether the input question can be answered with reasonable certainty based on the query library.
    """

    # Format the prompt with the actual questions
    formatted_query_library = "\n".join([f"- {q}" for q in query_library_questions])
    combined_prompt = question_filter_prompt.format(
        query_library_questions=formatted_query_library,
        input_question=question
    )

    # Get structured response
    response = await assistant.get_response(
        prompt=combined_prompt,
        response_schema=QuestionFilterResponse,
        quiet=True
    )

    return response


async def filter_questions_df(
        df: pd.DataFrame,
        question_column: str,
        query_library: Optional[QueryLibrary] = None,
        assistant_type: str = "gemini",
        model_name: str = "gemini-2.5-flash",
        max_workers: int = 50,
) -> pd.DataFrame:
    """
    Filter questions from a DataFrame in parallel to determine if they can be answered
    based on the query library.

    Args:
        df: Input DataFrame containing questions to process
        question_column: Name of the column containing the questions
        query_library: QueryLibrary object containing reference questions
        assistant_type: Type of LLM assistant to create
        model_name: Name of the model to use for the assistant
        max_workers: Maximum number of concurrent threads to use for processing

    Returns:
        DataFrame with added columns for filtering results:
        - is_answerable: Boolean indicating if the question can be answered
        - filter_rationale: Explanation of the filtering decision
        - suggested_questions: Alternative questions if not answerable
        - filter_time: Time taken for filtering in seconds
    """

    # Create a copy of the DataFrame
    result_df = df.copy()

    # Initialize filter result columns
    result_df['is_answerable'] = None
    result_df['filter_rationale'] = None
    result_df['suggested_questions'] = None
    result_df['filter_time'] = None

    # Get query library questions for context
    if query_library is not None:
        query_library_questions = query_library.df_querylib[query_library.col_question].tolist()
    else:
        # If no query library provided, use an empty list
        query_library_questions = []
        logger.warning("No query library provided. Filtering will be less effective.")

    # Use the actual DataFrame indices
    indices = df.index.tolist()

    # Define a synchronous function to process a single question
    def process_question(idx, question):
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Create a new assistant instance for this thread
        thread_assistant = create_assistant(
            assistant_type=assistant_type,
            model_name=model_name
        )

        try:
            # Filter the question
            start_time = time.time()
            filter_result = loop.run_until_complete(
                filter_question(
                    question,
                    query_library_questions,
                    thread_assistant
                )
            )
            filter_time = time.time() - start_time

            return idx, {
                "is_answerable": filter_result.answerable,
                "filter_rationale": filter_result.rationale,
                "suggested_questions": filter_result.suggested_questions,
                "filter_time": filter_time
            }

        except Exception as e:
            logger.error(f"Error filtering question at index {idx}: {str(e)}")
            return idx, {
                "error": str(e),
                "is_answerable": None,
                "filter_rationale": f"Error: {str(e)}",
                "suggested_questions": [],
                "filter_time": None
            }
        finally:
            loop.close()

    # Process all questions using a thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the thread pool
        future_to_idx = {
            executor.submit(process_question, idx, df.at[idx, question_column]): idx
            for idx in indices
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            idx, result = future.result()

            if "error" in result:
                logger.warning(f"Error in filtering for index {idx}: {result['error']}")
                continue

            # Update filter results
            result_df.at[idx, 'is_answerable'] = result['is_answerable']
            result_df.at[idx, 'filter_rationale'] = result['filter_rationale']
            result_df.at[idx, 'suggested_questions'] = "|||".join(result['suggested_questions']) if result[
                'suggested_questions'] else ""
            result_df.at[idx, 'filter_time'] = result['filter_time']

            # Log summary
            answerable_status = "Answerable" if result['is_answerable'] else "Not answerable"
            suggestions_count = len(result['suggested_questions'])
            logger.info(f"Question at index {idx}: {answerable_status}, "
                        f"Suggestions: {suggestions_count}, "
                        f"Time: {result['filter_time']:.2f}s")

    return result_df


async def filter_questions_with_querylib(
        question: str,
        querylib_file: Union[str, Path],
        col_question: str = "QUESTION",
        assistant_type: str = "gemini",
        model_name: str = "gemini-2.5-flash",
        subset_ids: Optional[List[int]] = None,
        question_type: str = None
) -> QuestionFilterResponse:
    """
    Filter a question using the query library.

    Args:
        question: The question to filter
        querylib_file: Path to the query library database file
        col_question: Column name for questions in the query library
        assistant_type: Type of assistant to use
        model_name: Name of the model to use
        subset_ids: Optional list of subset IDs
        question_type: Type of question to filter by ("QA" or "COHORT_GENERATOR" or None)

    Returns:
        FilterResponse object containing the response data
    """
    # Load query library
    querylib = QueryLibraryManager.get_instance().load_querylib(querylib_file=querylib_file)
    logger.info(f"Query library loaded from {querylib_file}")

    df_questions = get_subset_df(querylib.df_querylib, subset_ids=subset_ids, id_col="ID", question_type=question_type)

    # Process the question
    logger.info(f"Question: {question}")
    response = await filter_question(
        question=question,
        query_library_questions=df_questions[col_question],
        assistant_type=assistant_type,
        model_name=model_name,
    )

    logger.info(f"Answerable: {response.answerable}")
    logger.info(f"Rationale: {response.rationale}")
    logger.info(f"Answerable score (1=answerable, 0=unanswerable): {response.answerable_score}")
    logger.info(f"Suggested questions: {response.suggested_questions}")

    return response


async def filter_questions_for_answerability(
        questions: List[str],
        querylib_file: Path
) -> List[Dict[str, Any]]:
    """Filter questions for answerability in parallel."""

    tasks = [
        filter_questions_with_querylib(question=q, querylib_file=querylib_file)
        for q in questions
    ]
    results = await asyncio.gather(*tasks)
    logger.info(f"Answerability results: {results}")
    return results
