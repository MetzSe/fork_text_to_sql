import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from snowflake.connector.cursor import SnowflakeCursor

from ascent_ai.models.inference.rag import RAGProcessor
from ascent_ai.db.snowflake_session import execute_query
from ascent_ai.external_calls import MedicalCoder
from ascent_ai.config.settings import settings
from ascent_ai.schemas.constants import CodingType

logger = logging.getLogger(__name__)


class MedicalSQLProcessor:

    def __init__(self, recommender=None, assistant=None):
        if recommender is None:
            # Use mock recommender when none provided
            self.recommender = MedicalCoder()
        else:
            self.recommender = recommender

        self.assistant = assistant
        self.database_concepts = {}
        self.concept_not_found = []

    def set_database_concepts(self, database_concepts: Dict) -> None:
        """
        Restore the state of the MedicalSQLProcessor from a dictionary.

        Args:
            database_concepts (Dict): Dictionary containing the database_concepts
        """
        self.database_concepts = database_concepts

    def get_or_create_database_concepts(self, database_name):
        """
        Get or create the concept lists for a specific database name.
        Args:
            database_name (str): The name of the database.

        Returns:
            dict: A dictionary containing lists for 'condition', 'procedure', 'drug', 'measurement', and 'drug_class'.
        """
        if database_name not in self.database_concepts:
            self.database_concepts[database_name] = {
                "condition": [],
                "procedure": [],
                "drug": [],
                "measurement": [],
                "drug_class": [],
                "observation": [],
                "visit": [],
                "provider": [],
            }
        return self.database_concepts[database_name]

    def restructure_data(self, original_data):
        """
        Restructures the original data into a new format.

        :param original_data: A dictionary containing the original data structure
        :return: A list of dictionaries with the restructured data
        """
        categories = ["condition", "drug", "drug_class", "measurement", "observation", "procedure", "visit"]

        return [
            {"category": category, "name": name, "value": values}
            for category in categories
            for item in original_data.get(category, [])
            for name, values in item.items()
        ]

    def process_and_restructure(self, database_name):
        """
        Processes the database concepts and restructures the data.

        :param database_name: The name of the database to process
        :return: A dictionary with the restructured data
        """
        original_data = self.get_or_create_database_concepts(database_name)
        return self.restructure_data(original_data)

    @staticmethod
    async def get_dictionaries(db_session, domain_id):
        # Get the database name from the session
        database_name = db_session.connection.database

        # Generate a cache key based on database name and domain_id
        # cache_key = create_cache_key(database_name, domain_id)

        # Get cached result
        # cached_result = await cache.get(cache_key)
        # if cached_result is not None:
        #     return cached_result

        # Template SQL
        query_template = """
            SELECT DISTINCT c.VOCABULARY_ID
            FROM {table_name} AS t
            JOIN concept AS c ON t.{column_name} = c.concept_id
            WHERE c.VOCABULARY_ID IS NOT NULL;
        """
        # domain mapping
        # visit domain is missing, have a look
        domain_mapping = {
            "condition": ("CONDITION_OCCURRENCE", "condition_source_concept_id"),
            "drug": ("DRUG_EXPOSURE", "drug_source_concept_id"),
            "drug_class": ("DRUG_EXPOSURE", "drug_source_concept_id"),
            "procedure": ("PROCEDURE_OCCURRENCE", "procedure_source_concept_id"),
            "measurement": ("MEASUREMENT", "measurement_source_concept_id"),
            "observation": ("OBSERVATION", "observation_source_concept_id"),
            "visit": ('"VISIT_OCCURRENCE"', "visit_source_concept_id"),
        }

        # Check if the domain_id is valid.
        if domain_id not in domain_mapping:
            raise ValueError(f"Invalid domain_id: {domain_id}")

        # Get the table and column names.
        table_name, column_name = domain_mapping[domain_id]
        # Format
        query = query_template.format(table_name=table_name, column_name=column_name)
        # Execute the query.
        result, columns = await execute_query(db_session, query)
        # Return a list of strings.
        result_list = [row[0] for row in result if row[0] not in ("None", None)]
        # Cache the result
        # await cache.set(cache_key, result_list)

        return result_list

    async def get_replacement_value(
        self,
        category: str,
        name: str,
        preferred_coding_system: CodingType = CodingType.STANDARD_CODING,
        db_session: Optional[SnowflakeCursor] = None
    ) -> Optional[Dict[str, Any]]:
        database_concepts = self.get_or_create_database_concepts_for_category(category, db_session)
        concept = self.find_concept_by_name(database_concepts, name)
        is_drug_class = category == "drug_class"

        if concept:
            return concept[name] if is_drug_class else concept

        return await self.fetch_and_store_new_concept(category, name, preferred_coding_system, db_session)

    def get_or_create_database_concepts_for_category(self, category: str, db_session) -> List[Dict[str, Any]]:
        database_name = db_session.connection.database
        return self.get_or_create_database_concepts(database_name)[category]

    @staticmethod
    def find_concept_by_name(database_concepts: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
        name_lower = name.lower()
        for concept_dict in database_concepts:
            for key, value in concept_dict.items():
                if key.lower() == name_lower:
                    return {key: value}
        return None

    async def fetch_and_store_new_concept(
        self, category: str, name: str, preferred_coding_system, db_session
    ) -> Dict[str, Any]:
        medical_coder_params = await self.build_medical_coder_params(category, name, preferred_coding_system, db_session)
        entity_codes = await self.get_codes(medical_coder_params)

        self.store_new_concept(name, category, entity_codes, db_session)
        return entity_codes

    async def build_medical_coder_params(self, category: str, name: str, preferred_coding_system, db_session) -> Dict[
        str, Any]:
        # Get the database name from the session
        database_name = db_session.connection.database

        # Set top_k based on category
        top_k = 10000 if category.lower() == "drug" else 500

        return {
            "query": name,
            "domain_ids": [category.title()],
            "top_k": top_k,
            "llm_filter": "chatgpt",
            "cosine_similarity": 0.7,
            "standard_concept": "S" if preferred_coding_system != CodingType.SOURCE_CODING else None,
            "vocabulary": await self.get_dictionaries(db_session,
                                                      category) if preferred_coding_system == CodingType.SOURCE_CODING else None,
            "database": database_name,
        }

    def store_new_concept(self, name: str, category: str, entity_codes: Dict[str, Any], db_session):
        database_concepts = self.get_or_create_database_concepts_for_category(category, db_session)
        is_drug_class = category == "drug_class"

        if is_drug_class:
            drug_concepts = self.get_or_create_database_concepts_for_category("drug", db_session)
            drug_concepts_list = [{key: value} for key, value in entity_codes.items()]
            database_concepts.append({name: entity_codes})
            drug_concepts.extend(drug_concepts_list)
        else:
            database_concepts.append(entity_codes)

    async def get_codes(self, medical_coder_params):
        return await self.recommender.get_medical_codes(**medical_coder_params)

    async def get_ids_from_names(self, match, preferred_coding_system, db_session):
        category, name = match
        name_lower = name.lower()

        replacement_value = await self.get_replacement_value(category, name_lower,
                                                             preferred_coding_system=preferred_coding_system,
                                                             db_session=db_session,
                                                             )

        if category == "drug_class":
            replacement_value = {name_lower: self.flatten_list(replacement_value)}

        return self.format_replacement_result(category, name_lower, replacement_value)

    @staticmethod
    def flatten_list(data: dict) -> list:
        """Flattens a dictionary of lists into a single list."""
        return [item for sublist in data.values() for item in sublist]

    def format_replacement_result(self, category, name, replacement_value):
        if len(replacement_value[name]) > 0 and "error" not in replacement_value[name]:
            concept_ids = [str(concept["CONCEPT_ID"]) for concept in replacement_value[name]]
            return ",".join(concept_ids)
        else:
            self.concept_not_found.append(name)
            raise ConceptNotFoundError(category, name)

    def get_concept_id_not_found(self):
        return self.concept_not_found

    async def fill_codes_in_sql(self, sql_text, preferred_coding_system, db_session=None):
        pattern = r"\[([a-zA-Z_]+)@([a-zA-Z0-9_/\-\(\)\'\\ ]+)\]"

        if preferred_coding_system == CodingType.SOURCE_CODING:
            # if preferred_coding_system == SOURCE_CODING replace condition_concept_id to condition_source_concept_id
            sql_text = self.replace_concept_id_to_source_concept_id(sql_text)

        matches = re.findall(pattern, str(sql_text))
        modified_sql = await self.get_codes_for_entity_type_value_pairs(matches, sql_text,
                                                                        preferred_coding_system=preferred_coding_system,
                                                                        db_session=db_session)

        if "NO_CONCEPT_IDS_FOUND" in modified_sql:
            return None

        return modified_sql

    async def post_process_sql_query(
            self,
            sql_text: str,
            max_retries: int = 5,
            preferred_coding_system: CodingType = CodingType.STANDARD_CODING,
            rag_agent: Optional[RAGProcessor] = None,
            db_session: Optional[SnowflakeCursor] = None,
            custom_code_mappings: Optional[Dict[Tuple[str, str], str]] = None
    ) -> str:
        """ for docs on custom_code_mappings see get_codes_for_entity_type_value_pairs"""
        if not sql_text:
            logger.error("SQL text is empty.  Cannot process.")
            raise EmptySQLTextError("SQL text cannot be empty.")

        # # Pattern matches any text that follows after @ (e.g., from "[condition@atopic dermatitis]"
        pattern = r"\[([a-zA-Z_]+)@([a-zA-Z0-9_/\-\(\)\'\\ ]+)\]"
        attempts = 0
        current_sql = sql_text
        last_error = None

        while attempts <= max_retries:
            try:
                if preferred_coding_system == CodingType.SOURCE_CODING:
                    # if preferred_coding_system == CodingType.SOURCE_CODING replace *_concept_id to *_source_concept_id
                    sql_text = self.replace_concept_id_to_source_concept_id(sql_text)
                    logger.info("Replaced XXX_concept_id to XXX_source_concept_id for SOURCE_CODING")

                matches = list(set(re.findall(pattern, str(sql_text))))

                # Also lowercase the values in custom_code_mappings if it exists
                if custom_code_mappings:
                    custom_code_mappings = {
                        (entity_type, value.lower()): code
                        for (entity_type, value), code in custom_code_mappings.items()
                    }

                # list of tuples with entity type and value
                # example: matches = [('condition', 'acute ischemic stroke')]
                modified_sql = await self.get_codes_for_entity_type_value_pairs(
                    matches,
                    sql_text,
                    preferred_coding_system=preferred_coding_system,
                    db_session=db_session,
                    custom_code_mappings=custom_code_mappings
                )

                # Check if there are any concept_name in text inside the query
                if not self.is_sql_for_concept_name_in(sql_text):
                    # this is the successful execution
                    return modified_sql

                logger.info("Generated query found containing a CONCEPT ==. Self-healing...")

                # Make the sql correct
                current_sql = await self.handle_invalid_sql(rag_agent)
                attempts += 1

            except Exception as e:
                last_error = e
                if attempts >= max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded: {str(e)}")
                    raise last_error
                logger.warning(f"Attempt {attempts + 1} failed: {str(e)}")
                attempts += 1

        # If we've exhausted all retries and haven't returned yet
        if last_error:
            raise last_error
        return current_sql

    async def get_codes_for_entity_type_value_pairs(
            self,
            matches: List[Tuple[str, str]],
            sql_text: str,
            preferred_coding_system: CodingType = CodingType.STANDARD_CODING,
            db_session: Optional[SnowflakeCursor] = None,
            custom_code_mappings: Optional[Dict[Tuple[str, str], str]] = None
    ) -> str:
        """
        Process all regex matches and replace them in the SQL text.
        Allows for custom code mappings to override default lookups.

        Args:
            matches: List of tuples containing (entity_type, value) pairs
            sql_text: Original SQL query with placeholders
            preferred_coding_system: The coding system to use for lookups
            db_session: Database session for queries
            custom_code_mappings: Optional dictionary mapping (entity_type, value) to code strings
                                 Example:         custom_code_mappings = {
            ('condition', 'dysphagia'): '4310996,4159140',
            ('condition', 'atopic dermatitis'): '1112807, 1112808',
            ('procedure', 'appendectomy'): '2211444,2211445'
        }

        Returns:
            str: Modified SQL query with replacements
        """
        # Create a list to store coroutines for async lookups
        coroutines = []
        replacements = []
        custom_mapping_indices = []

        # Process each match
        for i, match in enumerate(matches):
            # Check if we have a custom mapping for this match
            if custom_code_mappings and match in custom_code_mappings:
                # Mark this index as having a custom mapping
                custom_mapping_indices.append(i)
                # Add a placeholder for this position
                coroutines.append(None)
                # Log the use of custom mapping
                logger.info(f"Using custom mapping for {match[0]}@{match[1]}: {custom_code_mappings[match]}")
            else:
                # Use the default lookup mechanism
                coroutines.append(
                    self.get_ids_from_names(
                        match,
                        preferred_coding_system=preferred_coding_system,
                        db_session=db_session,
                    )
                )

        # Execute all non-custom lookups asynchronously
        non_custom_results = await asyncio.gather(*[coro for coro in coroutines if coro is not None])

        # Merge custom mappings with async results
        result_index = 0
        for i in range(len(matches)):
            if i in custom_mapping_indices:
                # Use the custom mapping
                match = matches[i]
                replacements.append(custom_code_mappings[match])
            else:
                # Use the result from async lookup
                replacements.append(non_custom_results[result_index])
                result_index += 1

        return self.replace_placeholders(matches, replacements, sql_text)

    @staticmethod
    def replace_placeholders(matches, replacements, sql_text):
        """
        Apply the replacements to the SQL text.
        """
        # match = ('condition', 'acute ischemic stroke')
        # replacements = ['4310996,4159140,373503,43530669,43530670,4138327,609242,457661,4045755,4124838,42535112']

        modified_sql = sql_text

        for match, replacement in zip(matches, replacements):
            placeholder = f"[{match[0]}@{match[1]}]"

            # If replacement is a list of IDs, convert to comma-separated string
            # AZ: not sure why this is needed but I leave it here for now
            if isinstance(replacement, list):
                replacement_str = ", ".join(str(concept_id) for concept_id in replacement)
            else:
                replacement_str = replacement

            # Warn if empty replacement (will result in empty set)
            if replacement_str == "":
                logging.warning(f"No IDs found for {placeholder}, query will return empty set")

            # parenthesis around the ids are already included
            modified_sql = modified_sql.replace(placeholder, replacement_str)

        return modified_sql

    async def handle_invalid_sql(self, rag: RAGProcessor):
        """
        Handle the case where the SQL does not meet the required criteria.
        """
        prompt = """Your generated SQL query doesn't meet the requirements.
            Please correct the sql query based on the previously given instructions.
            [!IMPORTANT] Do not include conditions, such as 'WHERE concept_name IN ...' or  'WHERE concept_name = "..."'
            [!IMPORTANT] Do not return anything except the sql query"""

        rag.default_assistant.add_message(role="user", message=prompt)
        completed_prompt = await self.assistant.get_response()
        sql_text = self.parse_sql_from_response(completed_prompt)
        return sql_text

    @staticmethod
    def parse_sql_from_response(resp=""):
        if resp is None:
            resp = ""

        # Try standard patterns first
        pattern1 = r"(?:Snowflake )?SQL query:\s*\n\n(.*?);"
        # Pattern 2 should require closing ```
        pattern2 = r"(?:```sql|```)\s*([\s\S]*?)\s*```"

        # Fallback pattern - just look for content after ```sql
        pattern3 = r"```sql\s*([\s\S]*$)"

        match1 = re.search(pattern1, resp, re.DOTALL)
        match2 = re.search(pattern2, resp, re.DOTALL)
        match3 = re.search(pattern3, resp, re.DOTALL)

        if match2:
            logger.info("Found properly formatted SQL with markdown tags")
            return match2.group(1)
        elif match1:
            logger.info("Found SQL query with explicit 'SQL query:' prefix. Adding ; at the end")
            return match1.group(1) + ";"
        elif match3:
            logger.warning("Found SQL but missing proper closing tags - using fallback pattern")
            return match3.group(1)
        else:
            logger.error(f"No SQL found in response. Response length: {len(resp)}")
            raise NoSQLFoundError("No SQL code found in the response", response=resp)

    @staticmethod
    def parse_python_from_response(resp=""):
        pattern = r"(?:```python|```) ?\n([\s\S]+?)\n```"
        match = re.search(pattern, resp)
        if match:
            return match.group(1)
        else:
            logger.info("No python code found.")

    @staticmethod
    def parse_json_from_response(resp=""):
        pattern = r"(?:```json|```) ?\n([\s\S]+?)\n```"
        match = re.search(pattern, resp)
        if match:
            return match.group(1)
        else:
            logger.info("No JSON code found.")

    @staticmethod
    def save_string_to_file(text="", filename="log.txt"):
        with open(filename, "a") as text_file:
            text_file.write("\n")
            if text:
                text_file.write(text)

    @staticmethod
    def is_sql_for_concept_name_in(sql_text):
        pattern = r"\b(?:\w+\s*\(\s*)?concept_name\s*(?:\)\s*)?\s*(?:=|IN|LIKE|ILIKE)\s*(?:\(?\s*'[^']+'(?:\s*,\s*'[^']+')*\s*\)?|'%[^']+%')"
        match = re.search(pattern, str(sql_text), re.IGNORECASE)
        return bool(match)

    @staticmethod
    def replace_concept_id_to_source_concept_id(sql_text):
        # this does not include visit on purpose, since for visit we use the omop codes
        pattern = re.compile(r"(condition|drug|drug_class|procedure|measurement|observation)_concept_id", re.IGNORECASE)

        # Replacement function
        def replacement(match):
            concept_type = match.group(1)  # Extract the concept type (e.g., "condition", "drug", etc.)
            if match.group().islower():
                return f"{concept_type}_source_concept_id"
            else:
                return f"{concept_type.upper()}_SOURCE_CONCEPT_ID"

        # Apply the replacement
        return pattern.sub(replacement, sql_text)


class NoSQLFoundError(Exception):
    """Custom exception raised when SQL code cannot be found in a response."""

    def __init__(self, message="No SQL code found", response=None):
        self.message = message
        self.response = response
        super().__init__(self.message)

    def __str__(self):
        """Custom string representation of the error that includes the response"""
        error_msg = f"Error: {self.message}\n"
        if self.response:
            error_msg += f"Response content:\n{self.response}"
        return error_msg


class NoConceptIDsFoundError(Exception):
    """Custom exception raised when no concept IDs are found."""

    def __init__(self, message="No concept IDs found", matches=None, sql_text=None):
        self.message = message
        self.matches = matches  # The matches that were attempted to be processed
        self.sql_text = sql_text # The original SQL text
        super().__init__(self.message)

    def __str__(self):
        """Custom string representation of the error that includes the matches and SQL text."""
        error_msg = f"Error: {self.message}\n"
        if self.matches:
            error_msg += f"Matches attempted:\n{self.matches}\n"
        if self.sql_text:
            error_msg += f"Original SQL Text:\n{self.sql_text}\n"
        return error_msg


class EmptySQLTextError(Exception):
    """Custom exception raised when the SQL text is empty."""
    pass


class ConceptNotFoundError(Exception):
    """Raised when no concepts are found for a given name."""
    def __init__(self, category: str, name: str):
        self.category = category
        self.name = name
        super().__init__(f"No concept IDs found for: {category}@{name}")
