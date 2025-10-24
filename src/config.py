"""
Configuration file for Netflix ML project
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data file paths
RAW_DATA_FILE = RAW_DATA_DIR / "netflix_titles.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "netflix_processed.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# PyTorch settings
DEVICE = "cuda"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

# Feature columns
NUMERIC_FEATURES = ["release_year"]
CATEGORICAL_FEATURES = ["type", "rating", "country"]
TEXT_FEATURES = ["title", "description"]

# Target column
TARGET_COLUMN = "rating"

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"