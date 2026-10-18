"""Settings in one place. Secrets come from .env, never from the code."""
import os
from pathlib import Path

from dotenv import load_dotenv

from llm import ModelSettings

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "8"))
TRACE_DIR = ROOT / "traces"


def get_api_key() -> str:
    key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not key or "paste-your-key" in key:
        raise SystemExit(
            "NVIDIA_API_KEY is missing. Copy .env.example to .env and paste your "
            "key in (it starts with nvapi-)."
        )
    return key


def model_settings() -> ModelSettings:
    return ModelSettings(
        temperature=float(os.getenv("NVIDIA_TEMPERATURE", "1.0")),
        top_p=float(os.getenv("NVIDIA_TOP_P", "0.95")),
        max_tokens=int(os.getenv("NVIDIA_MAX_TOKENS", "16384")),
        thinking=os.getenv("NVIDIA_THINKING", "on").strip().lower(),
        retries=int(os.getenv("NVIDIA_RETRIES", "4")),
    )
