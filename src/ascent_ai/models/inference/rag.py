import logging

from ascent_ai.data.queries import querylib_loader
from ascent_ai.models.inference.assistants import create_assistant
from ascent_ai.data.queries.query_library import QueryLibrary

logger = logging.getLogger(__name__)

import logging
import threading
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


class QueryLibraryManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.querylib = None
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = cls()
        return cls._instance

    def load_querylib(self, main_path: Optional[str] = None, querylib_file: Optional[str] = None, embedding_model: str = "BAAI/bge-large-en-v1.5"):
        if self.querylib is not None:
            return self.querylib

        main_path = main_path or os.getcwd()
        querylib_file = querylib_file or querylib_loader.get_latest_querylib_file(main_path)

        querylib = QueryLibrary(
            querylib_name="patient_counts",
            source="gold_label_dec_2023",
            querylib_source_file=None,
            col_question="QUESTION",
            col_question_masked="QUESTION_MASKED",
            col_query_w_placeholders="QUERY_SNOWFLAKE_WITH_PLACEHOLDERS",
            col_query_executable="QUERY_SNOWFLAKE_RUNNABLE",
        )
        querylib = querylib.load(querylib_file=querylib_file)
        querylib.load_embedding_model(embedding_model_name=embedding_model)

        self.logger.info(f"Embedding loaded from {querylib_file}")
        self.querylib = querylib
        return self.querylib


@dataclass
class RAGConfig:
    main_path: Optional[Path] = None
    log_folder: Optional[Path] = None
    querylib_file: Optional[Path] = None
    assistant_type: str = "gpt"
    model_name: Optional[str] = None
    top_k_prompt: int = 5
    top_k_screening: int = 50
    sim_threshold: float = 0.0
    prompt_template: str = None
    database: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)


class AssistantState:
    """
    Represents the state of an assistant, including the main assistant
    and optionally assistant_answers.
    """

    def __init__(self, assistant, assistant_answers=None):
        self.assistant = assistant
        self.assistant_answers = assistant_answers


class RAGProcessor:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.querylib_manager = QueryLibraryManager.get_instance()

        # Default state for when no specific database is provided
        self.default_state = self._create_state()

        # Dictionary to store states for different databases
        self.database_states = {}

    @property
    def default_assistant(self):
        return self.default_state.assistant

    @property
    def default_assistant_answers(self):
        return self.default_state.assistant_answers

    @property
    def querylib(self):
        return self.querylib_manager.querylib

    def _create_state(self) -> AssistantState:
        """
        Create an AssistantState object based on the current configuration.
        """
        if self.config.assistant_type == "huggingface_custom_model":
            if not self.config.additional_params.get("model_path") or not self.config.additional_params.get("run_name"):
                raise ValueError("model_path and run_name must be provided for huggingface_custom assistant")
            assistant = create_assistant(
                assistant_type=self.config.assistant_type,
                model_path=self.config.additional_params.get("model_path"),
                run_name=self.config.additional_params.get("run_name"),
            )
            return AssistantState(assistant)
        else:
            assistant = create_assistant(
                assistant_type=self.config.assistant_type,
                model_name=self.config.model_name,
            )
            assistant_answers = create_assistant(
                assistant_type=self.config.assistant_type,
                model_name=self.config.model_name,
            )
            return AssistantState(assistant, assistant_answers)

    def get_state_for_database(self, database_name: Optional[str] = None) -> "RAGProcessor":
        """
        Retrieve a copy of the RAGProcessor for a specific database, or return the default processor.

        Args:
            database_name (Optional[str]): Name of the database.
                                           If None, returns the default RAGProcessor.

        Returns:
            RAGProcessor: A copy of the RAGProcessor configured for the specified database.
        """
        if database_name is None:
            return self

        # If a copy for the database already exists
        if database_name not in self.database_states:
            logger.info(f"Creating a copied RAGProcessor for database: {database_name}")

            # Create a new RAGProcessor instance
            database_processor = RAGProcessor(self.config)

            # Copy properties
            database_processor.querylib_manager = self.querylib_manager
            database_processor.default_state = self.default_state
            database_processor.database_states = self.database_states

            # Update the configuration for the specific database
            database_processor.config.database = database_name

            self.database_states[database_name] = database_processor

        return self.database_states[database_name]

    def get_state(self) -> Dict:
        """
        Get the current state of the RAGProcessor.
        Returns a dictionary containing serializable state data.
        """

        # Serialize the default state
        return {
            "default_state": self.default_state,
        }

    def set_state(self, state):
        """
        Get the current state of the RAGProcessor.
        Returns a dictionary containing serializable state data.
        """
        # Serialize the default state
        self.default_state = state.get("default_assistant", self._create_state())
