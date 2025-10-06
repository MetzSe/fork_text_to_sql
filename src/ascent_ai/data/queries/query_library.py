# coding=utf-8
__author__ = "Angelo Ziletti"
__maintainer__ = "Angelo Ziletti"

import asyncio
import logging
import os.path
from concurrent.futures import as_completed, ThreadPoolExecutor
from functools import cache
from typing import Optional
from datetime import datetime, date
import sqlite3
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ascent_ai.models.inference.prompts import entity_masking

tqdm.pandas()

logger = logging.getLogger(__name__)


@cache
def _import_sklearn_preprocessing():
    import sklearn.preprocessing as pp

    return pp


def normalize(*args, **kwargs):
    """sklearn.preprocessing.normalize with import caching"""
    pp = _import_sklearn_preprocessing()
    return pp.normalize(*args, **kwargs)


def make_label_encoder(*args, **kwargs):
    """sklearn.preprocessing.LabelEncoder with import caching"""
    pp = _import_sklearn_preprocessing()
    return pp.LabelEncoder(*args, **kwargs)


@cache
def _import_sentence_transformer() -> type:
    from sentence_transformers import SentenceTransformer as cls

    return cls


def make_sentence_transformer(*args, **kwargs):
    """sentence_transformers.SentenceTransformer with import caching"""
    cls = _import_sentence_transformer()
    return cls(*args, **kwargs)


class QueryLibrary:
    """Collection of queries for retrieval augmented generation"""

    def __init__(
        self,
        querylib_name: str,
        source: str,
        querylib_source_file: object,
        col_question: str,
        col_question_masked: str,
        col_query_w_placeholders: str,
        col_query_executable: Optional[str] = None,
        date_live: Optional[date] = None,
        storage_type: str = "sqlite"  # Default to sqlite only
    ) -> None:
        self.querylib_name = querylib_name
        self.date_live = date_live
        self.source = source
        self.col_question = col_question
        self.col_question_masked = col_question_masked
        self.col_query_w_placeholders = col_query_w_placeholders
        self.col_query_executable = col_query_executable
        self.storage_type = storage_type

        if querylib_source_file:
            df_querylib = pd.read_excel(querylib_source_file)
            self.df_querylib = df_querylib
        else:
            self.df_querylib = pd.DataFrame()
        self.label_encoder = None
        self.label_encoder_dict = None

        self.embeddings = []

        self.embedding_model = None

    @staticmethod
    def _init_sqlite_db(db_path: str) -> None:
        """Initialize SQLite database schema"""
        with sqlite3.connect(db_path) as conn:
            # Drop existing tables if they exist
            conn.execute("DROP TABLE IF EXISTS queries")
            conn.execute("DROP TABLE IF EXISTS embeddings")
            conn.execute("DROP TABLE IF EXISTS metadata")
            conn.execute("DROP TABLE IF EXISTS label_encoder")

            # Create queries table
            conn.execute("""
                CREATE TABLE queries (
                    id INTEGER PRIMARY KEY,
                    question TEXT NOT NULL,
                    question_masked TEXT,
                    query_with_placeholders TEXT,
                    query_executable TEXT,
                    question_type TEXT
                )
            """)

            # Create embeddings table with matrix_shape column
            conn.execute("""
                CREATE TABLE embeddings (
                    id INTEGER PRIMARY KEY,
                    model_name TEXT,
                    embed_matrix BLOB,
                    matrix_shape TEXT
                )
            """)

            # Create metadata table
            conn.execute("""
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value BLOB
                )
            """)

            # Create table for label encoder if needed
            conn.execute("""
                CREATE TABLE label_encoder (
                    id INTEGER PRIMARY KEY,
                    class_name TEXT,
                    encoded_value INTEGER
                )
            """)

            # Add indices
            conn.execute("CREATE INDEX idx_question_type ON queries(question_type)")
            conn.execute("CREATE INDEX idx_model_name ON embeddings(model_name)")

            conn.commit()

    def __len__(self):
        return len(self.df_querylib)

    def calc_embedding(self, embedding_model_name="BAAI/bge-large-en-v1.5", use_masked=True):
        """
        Calculate embeddings for the query library using the specified model.

        Args:
            embedding_model_name (str): Name of the embedding model to use
            use_masked (bool): Whether to use masked questions or original questions
        """
        logger.info(f"Starting embedding calculation with model: {embedding_model_name}")

        # Initialize embedding model
        embedding_model = make_sentence_transformer(embedding_model_name)

        # Select appropriate column
        if use_masked:
            col_txt = self.col_question_masked
            logger.info("Using masked questions for embedding")
        else:
            col_txt = self.col_question
            logger.info("Using original questions for embedding")

        # Log initial state
        initial_count = len(self.df_querylib)
        initial_null_count = self.df_querylib[col_txt].isna().sum()
        logger.info(f"Initial dataset state:")
        logger.info(f"- Total rows: {initial_count}")
        logger.info(f"- Null values: {initial_null_count}")

        # Clean data
        self.df_querylib = self.df_querylib.dropna(subset=[col_txt])

        # Log cleaning results
        clean_count = len(self.df_querylib)
        if initial_count != clean_count:
            logger.warning(f"Removed {initial_count - clean_count} rows with null values")
            logger.warning(f"Remaining rows: {clean_count}")

        # Ensure string type
        self.df_querylib[col_txt] = self.df_querylib[col_txt].astype(str)

        # Log embedding process start
        logger.info("Starting embedding calculation...")

        try:
            # Calculate embeddings
            embed_series = self.df_querylib[col_txt].progress_apply(
                lambda x: embedding_model.encode(x, normalize_embeddings=True)
            )

            # Convert to matrix
            embed_matrix = np.stack(embed_series.values)

            # Create embedding dictionary
            embedding = {
                "model_name": str(embedding_model_name),
                "embed_matrix": embed_matrix,
            }

            # Add to embeddings list
            self.embeddings.append(embedding)

            # Log success and details
            logger.info("Embedding calculation completed successfully")
            logger.info(f"Embedding matrix shape: {embed_matrix.shape}")
            logger.info(f"Number of embeddings: {len(embed_series)}")

            # Optional: Log some basic statistics about the embeddings
            avg_norm = np.mean(np.linalg.norm(embed_matrix, axis=1))
            logger.info(f"Average embedding vector norm: {avg_norm:.4f}")

        except Exception as e:
            logger.error(f"Error during embedding calculation: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            raise

        return embedding

    def save(self, querylib_file: str) -> None:
        """Save the query library to SQLite"""
        if not self.verify_embeddings():
            raise ValueError("Embedding verification failed")

        # Ensure file has .db extension
        if not querylib_file.endswith('.db'):
            querylib_file += '.db'
            logger.info(f"Added .db extension to file: {querylib_file}")

        self._init_sqlite_db(querylib_file)
        with sqlite3.connect(querylib_file) as conn:
            # Save metadata about the QueryLibrary instance (excluding binary objects for security)
            metadata = {
                'querylib_name': self.querylib_name,
                'source': self.source,
                'col_question': self.col_question,
                'col_question_masked': self.col_question_masked,
                'col_query_w_placeholders': self.col_query_w_placeholders,
                'col_query_executable': self.col_query_executable,
                'date_live': self.date_live.isoformat() if self.date_live else None,
                'storage_type': 'sqlite'
            }

            # Save metadata
            for key, value in metadata.items():
                conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                             (key, value))

            # Save queries
            self.df_querylib.to_sql('queries', conn, if_exists='replace', index=True)

            # Save embeddings with shape information
            if self.embeddings:
                for embedding in self.embeddings:
                    embed_matrix = embedding["embed_matrix"]
                    conn.execute(
                        """
                        INSERT INTO embeddings 
                        (model_name, embed_matrix, matrix_shape) 
                        VALUES (?, ?, ?)
                        """,
                        (
                            embedding["model_name"],
                            sqlite3.Binary(embed_matrix.tobytes()),
                            str(embed_matrix.shape)
                        )
                    )
            conn.commit()
            logger.info(f"Saved query library to SQLite database: {querylib_file}")

    def load_embedding_model(self, embedding_model_name):
        self.embedding_model = make_sentence_transformer(embedding_model_name)

    def verify_embeddings(self):
        """Verify the integrity of embeddings before saving"""
        if not self.embeddings:
            logger.warning("No embeddings found to verify")
            return False

        for idx, embedding in enumerate(self.embeddings):
            if 'model_name' not in embedding:
                logger.error(f"Embedding {idx} missing model_name")
                return False
            if 'embed_matrix' not in embedding:
                logger.error(f"Embedding {idx} missing embed_matrix")
                return False

            matrix = embedding['embed_matrix']
            if not isinstance(matrix, np.ndarray):
                logger.error(f"Embedding {idx} matrix is not a numpy array")
                return False

            logger.info(f"Embedding {idx} verified: {matrix.shape}")
        return True

    @staticmethod
    def load(querylib_file: Union[str, Path]):
        """Load the query library from SQLite"""
        logger.info(f"Loading query library from: {querylib_file}")

        # if the input is a string, convert it to path to correctly detect the storage format
        querylib_file = Path(querylib_file) if isinstance(querylib_file, str) else querylib_file

        if querylib_file.suffix.lower() == '.db':
            try:
                with sqlite3.connect(str(querylib_file)) as conn:
                    # Load metadata
                    metadata = dict(conn.execute("SELECT key, value FROM metadata").fetchall())

                    # Create new instance with metadata
                    query_lib = QueryLibrary(
                        querylib_name=metadata['querylib_name'],
                        source=metadata['source'],
                        querylib_source_file=None,  # We're loading from DB
                        col_question=metadata['col_question'],
                        col_question_masked=metadata['col_question_masked'],
                        col_query_w_placeholders=metadata['col_query_w_placeholders'],
                        col_query_executable=metadata.get('col_query_executable'),
                        date_live=datetime.fromisoformat(metadata['date_live'].decode()) if metadata[
                            'date_live'] else None,
                        storage_type="sqlite"
                    )

                    # Load binary objects - disabled for security reasons
                    # These fields will remain None to avoid pickle deserialization
                    query_lib.label_encoder = None
                    query_lib.label_encoder_dict = None
                    query_lib.embedding_model = None

                    if metadata.get('label_encoder'):
                        logger.warning("Label encoder found in database but skipped for security reasons")
                    if metadata.get('label_encoder_dict'):
                        logger.warning("Label encoder dict found in database but skipped for security reasons")
                    if metadata.get('embedding_model'):
                        logger.warning("Embedding model found in database but skipped for security reasons")

                    # Load queries
                    query_lib.df_querylib = pd.read_sql("SELECT * FROM queries", conn)

                    # Load embeddings
                    query_lib.embeddings = []
                    for model_name, embed_matrix, matrix_shape in conn.execute(
                            "SELECT model_name, embed_matrix, matrix_shape FROM embeddings"
                    ).fetchall():
                        # Convert string shape back to tuple
                        shape = tuple(map(int, matrix_shape.strip('()').split(',')))
                        try:
                            # matrix = np.frombuffer(embed_matrix).reshape(shape)
                            matrix = np.frombuffer(embed_matrix, dtype=np.float32).reshape(shape)
                            query_lib.embeddings.append({
                                "model_name": model_name,
                                "embed_matrix": matrix
                            })
                            logger.debug(f"Loaded embedding matrix with shape: {shape}")
                        except ValueError as e:
                            logger.error(f"Error reshaping matrix: {e}")
                            logger.error(f"Matrix size: {len(embed_matrix)}, Attempted shape: {shape}")
                            raise

                    logger.info("Query library loaded from SQLite database")
                    return query_lib

            except Exception as e:
                logger.exception(f"Error loading SQLite database: {e}")
                return None

        else:
            raise ValueError(f"Unsupported file type: {querylib_file}")

    def extract_idx_records(self, values_to_extract, source_col):
        idx_records = self.df_querylib.index[self.df_querylib[source_col].isin(values_to_extract)].tolist()
        return idx_records

    def extract_embed_matrix(self, value_rows_to_extract, extract_col_name, embedding):
        # Find the rows in the ontology that match name_rows_to_extract
        idx_records = self.extract_idx_records(value_rows_to_extract, extract_col_name)

        # Get the corresponding embedding matrix
        embed_matrix = embedding["embed_matrix"][idx_records]

        # Get the names from the matrix so they match the embeddings
        value_rows_embed = self.df_querylib.loc[idx_records][extract_col_name].reset_index(drop=True)

        return embed_matrix, value_rows_embed

    def fit_label_encoder(self, col="CONCEPT_CODE"):
        le = make_label_encoder()
        encoded_concept_codes = le.fit_transform(self.df_querylib[col])
        # in this way, only unique classes are represented
        self.label_encoder = le
        self.label_encoder_dict = dict(
            zip(
                self.label_encoder.classes_,
                self.label_encoder.transform(self.label_encoder.classes_),
            )
        )

        return encoded_concept_codes

    def select_records_by_value(self, col_select, value_select):
        """Extract from the ontology only the rows with a given value in a column"""

        df_onto_selected = self.df_querylib[self.df_querylib[col_select] == value_select]

        return df_onto_selected

    @staticmethod
    def add_separator_to_input_entities(lst, sep="[SEP_P]"):
        joined_list = []
        for inner_list in lst:
            joined_list.append(f" {sep} ".join(inner_list))
        return joined_list

    def get_similar_questions(
        self,
        samples,
        top_k=5,
        sim_threshold=0.95,
        normalize_score=True,
        col_search=None,
        max_rows=1000,
        tmp_dir=None,
        export_txt=False,
        question_type=None,
    ):
        if col_search is None:
            col_search = self.col_question

        df_querylib_selected = self.df_querylib

        # Apply question type filter if specified
        if question_type is not None:
            if question_type not in ["QA", "COHORT_GENERATOR"]:
                raise ValueError("question_type must be either 'QA' or 'COHORT_GENERATOR'")
            df_querylib_selected = df_querylib_selected[df_querylib_selected["QUESTION_TYPE"] == question_type]

        embed_matrix, names_avail = self.extract_embed_matrix(
            value_rows_to_extract=df_querylib_selected[self.col_question].tolist(),
            extract_col_name=self.col_question,
            embedding=self.embeddings[0],
        )

        # Cast the input samples in a dataframe for convenience
        samples_with_sep = self.add_separator_to_input_entities(samples)
        df_input_names = pd.DataFrame(samples_with_sep, columns=[self.col_question])

        df_input_names_list = [df_input_names[i : i + max_rows] for i in range(0, df_input_names.shape[0], max_rows)]

        df_recap_recs_list = []
        df_recs_list = []

        # Temporary directory functionality is not supported for security reasons
        if tmp_dir is not None:
            logger.error("Temporary directory functionality is not supported for security reasons")
            raise NotImplementedError("Temporary directory functionality is not supported for security reasons")

        # Parallel processing for get_similar_names
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.get_similar_names,
                    df_input_chunk,
                    names_avail,
                    embed_matrix=embed_matrix,
                    embedding_model=self.embedding_model,
                    col_search=col_search,
                    normalize_score=normalize_score,
                    top_k=top_k,
                    sim_threshold=sim_threshold,
                    export_txt=export_txt,
                ): idx
                for idx, df_input_chunk in enumerate(df_input_names_list)
            }

            for future in as_completed(futures):
                idx = futures[future]
                df_recap_recs, df_recs = future.result()
                df_recap_recs_list.append(df_recap_recs)
                df_recs_list.extend(df_recs)


        df_recap_recs = pd.concat(df_recap_recs_list)

        return df_recap_recs, df_recs_list

    def get_similar_names(
        self,
        df,
        classes,
        embed_matrix,
        embedding_model,
        col_search,
        suffix="unsupervised",
        normalize_score=True,
        top_k=20,
        sim_threshold=0.0,
        top_k_limit=None,
        export_txt=True,
    ):

        if top_k_limit is None:
            top_k_limit = len(classes)

        logger.debug("Retrieving the most similar classes")

        # remove leading and trailing spaces
        df[col_search] = df[col_search].astype(str).str.strip()

        # Compute embeddings in batches (assuming embedding_model can handle batch input)
        text_embeddings = embedding_model.encode(df[self.col_question].tolist(), normalize_embeddings=True)

        logger.debug(f"Input embed_matrix shape: {embed_matrix.shape}")
        logger.debug(f"Text embeddings shape: {text_embeddings.shape}")

        logger.debug(f"Text embedding size: {text_embeddings.nbytes / 10 ** 6} (Mb)")
        logger.debug(f"text_embedding shape: {text_embeddings.shape}")

        if normalize_score:
            text_embeddings = normalize(text_embeddings)
            embed_matrix = normalize(embed_matrix)

        # Efficient matrix multiplication
        sim_matrix = text_embeddings @ embed_matrix.T

        logger.debug(f"Similarity matrix size: {sim_matrix.nbytes / 10 ** 6} (Mb)")
        logger.debug(f"sim_matrix shape: {sim_matrix.shape}")

        # Efficient top-k selection
        idx_match_sorted = np.argpartition(-sim_matrix, kth=top_k_limit - 1, axis=1)[:, :top_k_limit]

        matched_classes_list = []
        scores_list = []
        code_list = []
        df_class_score_list = []

        for idx_input in range(sim_matrix.shape[0]):
            idx_match_input = idx_match_sorted[idx_input]
            sim_matrix_row_sorted = -np.partition(-sim_matrix[idx_input], kth=top_k_limit - 1)[:top_k_limit]
            class_text_sorted = classes[idx_match_input]

            df_class_scores = pd.DataFrame(
                zip(class_text_sorted, sim_matrix_row_sorted),
                columns=["Class", "Score"],
            )
            df_class_scores = df_class_scores.nlargest(top_k, "Score")

            matched_classes_list.append(df_class_scores["Class"].tolist())
            scores_list.append(df_class_scores["Score"].tolist())

            df_class_scores["QUESTION"] = df_class_scores["Class"]
            code_list.append(df_class_scores["QUESTION"].tolist())

            if not export_txt:
                df_class_scores = df_class_scores.drop(["Class"], axis=1, errors="ignore")

            df_class_score_list.append(df_class_scores)

        df[f"rec_{suffix}_questions"] = code_list
        df[f"rec_{suffix}_scores"] = scores_list

        return df, df_class_score_list

    def get_df_recs(self, question, top_k, sim_threshold, question_type):
        df_recap_recs, df_recs_list = self.get_similar_questions(question, top_k=top_k, sim_threshold=sim_threshold, question_type=question_type)

        # here the list is only one element long because we pass only one question
        df_recs_list_merged = []
        for df_recs in df_recs_list:
            df_merged = df_recs.merge(self.df_querylib, on="QUESTION", how="left")
            df_recs_list_merged.append(df_merged)

        # here we only have one question thus the list is always one element
        # TO DO: this is not ideal and it should be improve later
        df_recs_list_out = df_recs_list_merged[0]
        return df_recs_list_out

    async def text_sql_template_for_rag(
        self,
        question_masked,
        top_k_screening,
        top_k_prompt,
        sim_threshold,
        reverse_order=False,
        rag_random=False,  # Parameter for random retrieval
        drop_first=False,  # Parameter to drop the first element
        question_type=None,
    ):
        # df_recs_list_out = self.get_df_recs([[question_masked]], top_k=top_k_screening, sim_threshold=sim_threshold, question_type=question_type)

        df_recs_list_out = await asyncio.to_thread(
            self.get_df_recs,
            [[question_masked]],
            top_k=top_k_screening,
            sim_threshold=sim_threshold,
            question_type=question_type
        )

        # If reverse_order is True, reverse the order of the DataFrame
        if reverse_order:
            df_recs_list_out = df_recs_list_out.sort_index(ascending=False)

        # If drop_first is True, drop the first element from the DataFrame
        if drop_first:
            logger.warning("Dropping first element of the retrieved queries")
            df_recs_list_out = df_recs_list_out.drop(df_recs_list_out.index[0])

        # If rag_random is True, randomly select one sample from the top-k
        if rag_random:
            logger.warning("Using random retrieval for RAG")
            df_recs_list_out = df_recs_list_out.sample(n=top_k_screening)

        # Keep only the top_k_prompt elements
        df_recs_list_out = df_recs_list_out.head(top_k_prompt)

        initial_sentence = "\n\nYou might find these example queries helpful: "

        # Modify the template based on question_type
        if question_type == "COHORT_GENERATOR":
            text_sql_template = (
                initial_sentence
                + "\n\n"
                + "\n\n".join(
                    f"#Example inclusion/exclusion criteria:\n{rec[self.col_question]}\n#Example SQL query:\n{rec[self.col_query_w_placeholders]}"
                    for rec in df_recs_list_out.to_dict("records")
                )
            )
        else:  # None or "QA"
            text_sql_template = (
                initial_sentence
                + "\n\n"
                + "\n\n".join(
                    f"#Example Question:\n{rec[self.col_question]}\n#Example SQL query:\n{rec[self.col_query_w_placeholders]}"
                    for rec in df_recs_list_out.to_dict("records")
                )
            )

        return text_sql_template, df_recs_list_out

    @staticmethod
    async def get_masked_question(question, assistant, reset_conversation=True, mask="DRUG_CLASS"):
        """
        :param prompts: List of prompts
        :param question: User question
        :param assistant: Assistant to use
        :param reset_conversation: True or False to reset the conversation
        :param mask: Mask to apply
        :return: masked question, question
        """
        prompt = entity_masking.format(question=question)

        assistant.add_message(role="user", message=prompt)
        masked_question = await assistant.get_response()

        if mask in masked_question:
            if masked_question.count(mask) > 1 or (masked_question.count(mask) == 1 and masked_question.count("DRUG") > 1):
                question += " Can you output also intermediate results for each drug class?"

        logger.info(f"Masked question: {masked_question}")

        if reset_conversation:
            assistant.reset_conversation()

        return masked_question, question
