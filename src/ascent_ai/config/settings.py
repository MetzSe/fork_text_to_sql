from pathlib import Path
from typing import Optional, Union, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os
import logging
from datetime import datetime
import sys

logger = logging.getLogger(__name__)

# At the top of settings.py, before any class definitions
if os.getenv("ENV_LOCAL_PATH"):
    BASE_DIR = Path(os.environ["ENV_LOCAL_PATH"]).resolve()
else:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Print for debugging
logger.debug(f"Using BASE_DIR: {BASE_DIR}")


def get_env_files(base_dir: Path) -> List[Path]:
    """Get environment files."""
    return [
        base_dir / ".env",
        base_dir / ".env.local",
        base_dir / ".env.dev",
        Path(".env"),
        Path(".env.local"),
    ]


def setup_environment(base_dir: Path) -> Path:
    """
    Setup the environment by loading environment variables into os.environ.
    """
    try:
        if not base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {base_dir}")

        logger.info(f"Base directory: {base_dir}")

        # Get env files
        env_files = get_env_files(base_dir)

        # Load environment variables with priority
        for env_file in env_files:
            if env_file.exists():
                load_dotenv(env_file, override=True)
                logger.info(f"Loaded environment variables from: {env_file}")
            else:
                logger.debug(f"Environment file not found (skipping): {env_file}")

        return base_dir

    except Exception as e:
        logger.error(f"Error setting up environment: {str(e)}")
        raise


ROOT_DIR = setup_environment(BASE_DIR)


class Settings(BaseSettings):
    """Type-safe settings management using Pydantic.
    Note: We also load variables via load_dotenv because some parts
    of the application require variables to be in os.environ.
    """

    model_config = SettingsConfigDict(
        # Use the same BASE_DIR for Pydantic's env file loading
        env_file=[str(f) for f in get_env_files(BASE_DIR)],
        extra="allow",
        case_sensitive=True,
    )

    BASE_DIR: Path = BASE_DIR  # Initialize with the BASE_DIR calculated earlier

    # Application settings
    RELEASE_VERSION: str = "local"
    CORS_ORIGINS: str = "*"
    DEBUG: bool = False

    # API endpoints
    MEDICAL_CODER_BASE_URL: Optional[str] = None

    # Snowflake configuration
    SNOWFLAKE_USER: Optional[str] = None
    SNOWFLAKE_PASSWORD: Optional[str] = None
    SNOWFLAKE_ACCOUNT_IDENTIFIER: Optional[str] = None
    SNOWFLAKE_WAREHOUSE: Optional[str] = None
    SNOWFLAKE_DATABASE: Optional[str] = None
    SNOWFLAKE_DATABASE_SCHEMA: Optional[str] = None
    SNOWFLAKE_TIMEOUT: int = 500

    # OpenAI configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_VERSION: Optional[str] = None
    OPENAI_API_BASE: Optional[str] = None
    MODEL_NAME: Optional[str] = None
    OPENAI_EAST_US_BASE: Optional[str] = None
    OPENAI_EAST_US_KEY: Optional[str] = None
    OPENAI_EAST_US_GPT4O_MODEL_NAME: Optional[str] = None
    OPENAI_EAST_US_GPT4O_API_VERSION: Optional[str] = None

    # Google configuration
    GOOGLE_API_KEY: Optional[str] = None

    # Logging
    LOG_LEVEL: str = "INFO"
    APP_VERSION: Optional[str] = None

    # Logging configuration
    LOG_BASE_DIR: Optional[str] = None
    LOG_SUBDIRS: bool = True
    LOG_SUBDIR_FORMAT: str = "%Y/%m/%d"  # Default to Year/Month structure

    # Logging configuration
    LOG_FORMAT: str = "%(asctime)s [%(levelname)s] - %(name)s: %(message)s"
    LOG_DATE_FORMAT: str = "%H:%M:%S"
    LOG_FILE_FORMAT: str = "%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s"
    SILENT_LOGGERS: dict[str, str] = {
        "snowflake.connector": "WARNING",
        "sentence_transformers.SentenceTransformer": "WARNING",
        "ascent.external_calls.ascent_client": "WARNING",
        "google_genai.models": "WARNING",
        "httpx": "WARNING",
    }

    @property
    def log_directory(self) -> Path:
        """
        Get the log directory with optional date-based subdirectories.
        If LOG_BASE_DIR is set in environment, uses that.
        Otherwise defaults to ROOT_DIR/logs.
        """
        if self.LOG_BASE_DIR:
            base = Path(self.LOG_BASE_DIR)
        else:
            base = ROOT_DIR / "logs"

        if self.LOG_SUBDIRS:
            # Create date-based subdirectory using configured format
            base = base / datetime.now().strftime(self.LOG_SUBDIR_FORMAT)

        # Ensure the directory exists
        base.mkdir(parents=True, exist_ok=True)
        return base

    _logging_initialized: bool = False

    def setup_logging(self, force: bool = False) -> None:
        """
        Setup logging configuration. Will only run once unless force=True.
        """
        if self._logging_initialized and not force:
            return

        # Basic console logging
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(self.LOG_FORMAT))

        logging.basicConfig(
            level=logging.DEBUG if self.DEBUG else logging.INFO,
            handlers=[handler],
            force=True,  # Ensure we can reset the basic config if needed
        )

        # File logging if log_directory is set
        log_file = (
            self.log_directory / f"app_{datetime.now().strftime('%Y%m%d%H%M%S_%f')}.log"
        )
        file_handler = logging.FileHandler(
            log_file,
            encoding="utf-8",  # Add UTF-8 encoding for file handler
        )
        file_handler.setFormatter(
            logging.Formatter(self.LOG_FILE_FORMAT, datefmt=self.LOG_DATE_FORMAT)
        )

        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        # Configure silent loggers
        for logger_name, level in self.SILENT_LOGGERS.items():
            logging.getLogger(logger_name).setLevel(getattr(logging, level))

        logger.info(f"Logging file saved at {log_file}")
        self._logging_initialized = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Automatically setup logging when Settings is instantiated
        self.setup_logging()


# Create the singleton instance
settings = Settings()
