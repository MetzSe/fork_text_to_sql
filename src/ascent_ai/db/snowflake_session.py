import asyncio
import logging
import random
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import snowflake.connector
from snowflake.connector import OperationalError, ProgrammingError, SnowflakeConnection
from snowflake.connector.constants import QueryStatus
from snowflake.connector.cursor import SnowflakeCursor
from sqlalchemy import create_engine

from ascent_ai.config.settings import settings
# from ascent.util.thread_pool import get_executor

logger = logging.getLogger(__name__)


class SnowflakeConnectionPool:
    def __init__(self, max_connections=15):
        self.max_connections = max_connections
        self.semaphore = asyncio.Semaphore(max_connections)
        self.connections: Dict[str, List[SnowflakeConnection]] = defaultdict(list)

    async def pre_warm_pool(self, db_names):
        for db_name in db_names:
            for _ in range(self.max_connections // len(db_names)):
                connection = await self._create_connection(db_name)
                self.connections[db_name].append(connection)

    async def get_connection(self, db_name: str) -> SnowflakeConnection:
        async with self.semaphore:
            if not self.connections[db_name]:
                return await self._create_connection(db_name)
            return self.connections[db_name].pop()

    async def _create_connection(self, db_name: str) -> SnowflakeConnection:
        connection_config = {
            "account": settings.SNOWFLAKE_ACCOUNT_IDENTIFIER,
            "user": settings.SNOWFLAKE_USER,
            "password": settings.SNOWFLAKE_PASSWORD,
            "warehouse": settings.SNOWFLAKE_WAREHOUSE,
            "database": db_name,
            "client_session_keep_alive": True,
            "client_session_keep_alive_heartbeat_frequency": 3600,
            "login_timeout": settings.SNOWFLAKE_TIMEOUT,
            "paramstyle": "qmark",
            "lower_case_identifiers": True,
            "session_parameters": {"STATEMENT_TIMEOUT_IN_SECONDS": settings.SNOWFLAKE_TIMEOUT, "DATE_FORMAT": "YYYY-MM-DD"},
        }
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: snowflake.connector.connect(**connection_config))

    def release_connection(self, db_name: str, connection: SnowflakeConnection):
        if len(self.connections[db_name]) < self.max_connections:
            self.connections[db_name].append(connection)
        else:
            connection.close()

    async def close_all_connections(self):
        for db_name, connections in self.connections.items():
            while connections:
                connection = connections.pop()
                await asyncio.to_thread(connection.close)
        self.connections.clear()
        logger.info("All database connections have been closed.")


connection_pool = SnowflakeConnectionPool()

schema_cache = {}


@asynccontextmanager
async def get_db(db_name: Optional[str] = None, schema_name: Optional[str] = None) -> AsyncGenerator:
    if not db_name:
        db_name = "OPTUM_CLAIMS_OMOP"

    connection = await connection_pool.get_connection(db_name)
    cursor = connection.cursor()

    try:
        if db_name:
            # Use the provided schema_name if available
            if not schema_name:
                if db_name == "ASCENT":
                    schema_name = "PUBLIC"
                else:
                    schema_name = await get_current_schema(cursor, db_name)

            logger.debug(f"Set {db_name}.{schema_name} as default")
            await execute_query(cursor, f'USE SCHEMA "{db_name}"."{schema_name}"')

        yield cursor

    except Exception as e:
        logger.exception(f"Error while connecting to the database: {e}")
        raise

    finally:
        cursor.close()
        connection_pool.release_connection(db_name, connection)


async def execute_query(cursor, query: str, params: Optional[List[Any]] = None, bulk: bool = False) -> Any:
    try:
        if params:
            if bulk:
                query_id = await asyncio.to_thread(cursor.executemany, query, params, _exec_async=True, _no_results=True)
                query_id = query_id.sfqid
            else:
                query_id = await asyncio.to_thread(cursor.execute_async, query, params)
                query_id = query_id.get("queryId")
        else:
            query_id = await asyncio.to_thread(cursor.execute_async, query)
            query_id = query_id.get("queryId")

        base_delay = 0.1
        max_delay = 3
        while True:
            status = await asyncio.to_thread(cursor.connection.get_query_status_throw_if_error, query_id)
            if status == QueryStatus.SUCCESS:
                await asyncio.to_thread(cursor.get_results_from_sfqid, query_id)
                results = await asyncio.to_thread(cursor.fetchall)
                columns = [meta.name.lower() for meta in cursor.description or ()]
                return results, columns
            elif status == QueryStatus.RUNNING:
                delay = min(base_delay * (1 + random.random()), max_delay)
                await asyncio.sleep(delay)
                base_delay *= 2
    except (ProgrammingError, OperationalError) as e:
        logger.error(f"Snowflake Error: {str(e)}")
        logger.error(f"Query: {query}")
        if params:
            logger.error(f"Parameters: {params}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Query: {query}")
        if params:
            logger.error(f"Parameters: {params}")
        raise


async def get_current_schema(cursor: SnowflakeCursor, database_name: str) -> str:
    if database_name in schema_cache:
        return schema_cache[database_name]

    await execute_query(cursor, f"show schemas in database {database_name};")
    result, _ = await execute_query(
        cursor,
        """
            select max("name") as schema_name
            from TABLE(RESULT_SCAN(LAST_QUERY_ID()))
            where "name" like 'CDM%' or "name" like 'DATA%'
    """,
    )
    schema_name = result[0][0] if result and result[0][0] else None

    if schema_name:
        schema_cache[database_name] = schema_name

    return schema_name

