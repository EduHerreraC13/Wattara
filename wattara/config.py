"""Central configuration: model settings, example tariff/emission values and API-key loading.

Every tunable number lives here, so it is easy to audit.
Environment variables (or a .env file) can override the model settings.
"""
import os

try:                                    # optional: load a local .env file if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ----------------------------- Meta Model API -----------------------------
API_BASE_URL = "https://api.meta.ai/v1"                 # OpenAI-compatible endpoint
MODEL = os.getenv("WATTARA_MODEL", "muse-spark-1.3")    # avoid the "-contributor" variants (see README)
PRICE_IN_PER_M = 1.25                                   # USD per million input tokens (Muse Spark, standard)
PRICE_OUT_PER_M = 4.25                                  # USD per million output tokens
BUDGET_USD = float(os.getenv("WATTARA_BUDGET_USD", "0.50"))      # local safety cap per session (an estimate)
AGENT_MAX_TOKENS = int(os.getenv("WATTARA_MAX_TOKENS", "3000"))  # reasoning tokens count as output!
MAX_STEPS = int(os.getenv("WATTARA_MAX_STEPS", "8"))    # tool-calling rounds before forcing a final answer
REASONING_EFFORT = os.getenv("WATTARA_REASONING_EFFORT") or None  # e.g. "low"; None = API default

# ------------------------------- Domain values -------------------------------
# EXAMPLE values: replace them with official figures and cite the source before claiming real savings.
TARIFF_PER_KWH = 0.70                  # soles (PEN) per kWh  -> example, not an official tariff
EMISSION_FACTOR_KG_PER_KWH = 0.20      # kg CO2 per kWh       -> example, not an official factor
MAX_CUT_MIN = 20                       # max minutes/day a single action may trim from a scheduled appliance


def get_api_key() -> str:
    """Return the Meta Model API key from the environment, a .env file or Colab secrets."""
    key = os.getenv("META_API_KEY")
    if key:
        return key
    try:                                # Google Colab: Secrets panel (key icon)
        from google.colab import userdata
        return userdata.get("META_API_KEY")
    except Exception:
        raise RuntimeError(
            "META_API_KEY not found. Copy .env.example to .env and paste your key "
            "(or add it as a Colab secret named META_API_KEY)."
        )
