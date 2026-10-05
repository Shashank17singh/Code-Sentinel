"""Central configuration: paths, model names and thresholds.

Everything can be overridden with environment variables (a local ``.env`` file is
loaded automatically) so the same code runs locally, in CI and on Streamlit Cloud.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("CODESENTINEL_DATA_DIR", ROOT / "data"))
PROBLEMS_DIR = DATA_DIR / "problems"
EVALS_DIR = DATA_DIR / "evals"
SAMPLES_DIR = DATA_DIR / "samples"
REVIEWS_FILE = Path(os.getenv("CODESENTINEL_REVIEWS_PATH", DATA_DIR / "reviews.json"))

# --- LLM (Groq) -------------------------------------------------------------
DEFAULT_MODEL = os.getenv("CODESENTINEL_MODEL", "openai/gpt-oss-120b")
AVAILABLE_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]
# Models used to generate "reference" AI solutions. Variety matters: every model has its own style.
REFERENCE_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
REFERENCE_RUNS_PER_MODEL = 3
REFERENCE_TEMPERATURE = 0.8
AGENT_MAX_ATTEMPTS = 5

# --- Detection thresholds ---------------------------------------------------
KGRAM_SIZE = 5  # tokens per k-gram
WINNOW_WINDOW = 4  # hashes per winnowing window
COMMON_THRESHOLD = 4  # fingerprints shared by more candidates than this are "common code"
SIMILARITY_FLAG_AT = 0.5  # classmate similarity that triggers a flag
REFERENCE_FLAG_AT = 0.5  # share of a submission found in an AI reference that triggers a flag
STYLE_FLAG_AT = 0.6  # heuristic / LLM style score that triggers a flag

# --- Sandbox ----------------------------------------------------------------
RUN_TIMEOUT_SECONDS = 5
RUN_MEMORY_MB = 512
RUN_MAX_OUTPUT_CHARS = 10_000
