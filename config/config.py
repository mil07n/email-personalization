"""
Central Configuration for NLP-Powered Executive Email Assistant.
Path constants and operational parameters.
"""

from pathlib import Path

# Base directories
CONFIG_DIR = Path(__file__).parent.resolve()
BASE_DIR = CONFIG_DIR.parent.resolve()

DATASETS_DIR = BASE_DIR / "datasets"
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
EVALUATION_DIR = BASE_DIR / "evaluation"
TESTS_DIR = BASE_DIR / "tests"

# Dataset directory paths
ENRON_DIR = DATASETS_DIR / "enron"
ENRON_RAW = ENRON_DIR / "raw"
ENRON_PROCESSED = ENRON_DIR / "processed"
ENRON_CSV = ENRON_PROCESSED / "aeslc.csv"

CUSTOMER_CARE_DIR = DATASETS_DIR / "customer_care"
CUSTOMER_CARE_RAW = CUSTOMER_CARE_DIR / "raw"
CUSTOMER_CARE_PROCESSED = CUSTOMER_CARE_DIR / "processed"
CUSTOMER_CARE_CSV = CUSTOMER_CARE_PROCESSED / "customer_care_emails.csv"

EMAIL_THREAD_SUMMARY_DIR = DATASETS_DIR / "email_thread_summary"
EMAIL_THREAD_SUMMARY_RAW = EMAIL_THREAD_SUMMARY_DIR / "raw"
EMAIL_THREAD_SUMMARY_PROCESSED = EMAIL_THREAD_SUMMARY_DIR / "processed"
EMAIL_THREAD_SUMMARY_CSV = EMAIL_THREAD_SUMMARY_PROCESSED / "email_thread_summary.csv"

EPA_DIR = DATASETS_DIR / "epa"
EPA_RAW = EPA_DIR / "raw"
EPA_PROCESSED = EPA_DIR / "processed"
EPA_CSV = EPA_PROCESSED / "epa_tasks.csv"

EMAILSUM_DIR = DATASETS_DIR / "emailsum"
EMAILSUM_RAW = EMAILSUM_DIR / "raw"
EMAILSUM_PROCESSED = EMAILSUM_DIR / "processed"
EMAILSUM_CSV = EMAILSUM_PROCESSED / "emailsum.csv"

# Trained model paths
URGENCY_MODEL_PATH = MODELS_DIR / "urgency_model.joblib"
URGENCY_VECTORIZER_PATH = MODELS_DIR / "urgency_vectorizer.joblib"
URGENCY_METRICS_PATH = EVALUATION_DIR / "urgency_metrics.json"

# Summarization evaluation
SUMMARIZATION_METRICS_PATH = EVALUATION_DIR / "summarization_metrics.json"

# User preferences
USER_PREFS_PATH = CONFIG_DIR / "user_preferences.json"
