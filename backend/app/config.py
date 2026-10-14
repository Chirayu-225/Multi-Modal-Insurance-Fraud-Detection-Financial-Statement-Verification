"""
Central configuration. Everything that differs between "runs with zero
external dependencies" (the default, for development and demos) and
"runs against the real vision API" (the production-ish path) is a flag
here, read from environment variables so nothing secret ever gets
committed to the repo.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- database ---
    # Async URL used by the app at runtime (asyncpg driver).
    database_url: str = "postgresql+asyncpg://claims:claims@db:5432/claims"

    # --- vision / OCR backend selection ---
    # "mock"   -> deterministic-but-randomized fake extraction, no network calls,
    #             lets the whole pipeline run with zero setup.
    # "gemini" -> real calls to the Gemini API (needs GEMINI_API_KEY).
    vision_backend: str = "mock"
    gemini_api_key: str | None = None
    # gemini-2.0-flash (the original default) was scheduled for shutdown
    # in June 2026. "gemini-flash-latest" is Google's moving alias for the
    # current Flash model; pin a specific model ID instead once you've
    # checked in AI Studio which Flash models your free tier covers.
    gemini_model: str = "gemini-flash-latest"

    # --- hosted LLM gateway (services/llm_gateway.py) ---
    # Order of providers tried for each task type. "mock" is offline and
    # deterministic, so the app runs with zero keys; real providers are
    # skipped automatically when their key is empty.
    llm_text_providers: str = "groq,openrouter,mock"
    llm_vision_providers: str = "gemini,mock"
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    openrouter_api_key: str | None = None
    # Pick any model with a ":free" suffix from OpenRouter's model list;
    # the free roster changes often, so there is no safe default.
    openrouter_model: str | None = None
    # Optional local fallback via Ollama's OpenAI-compatible endpoint,
    # e.g. "http://localhost:11434/v1" with model "qwen3:4b".
    local_llm_base_url: str | None = None
    local_llm_model: str = "qwen3:4b"
    llm_cache_path: str = "data/llm_cache.sqlite3"
    llm_max_retries: int = 2

    # --- privacy guard (services/privacy_guard.py) ---
    # Never set this to true with real claimant or applicant data: it lets
    # restricted content reach hosted free-tier APIs. It exists only so a
    # developer can deliberately test the guard's failure mode.
    allow_restricted_to_hosted: bool = False

    # --- consistency engine selection ---
    # "rules"  -> hand-written rules engine (services/consistency.py)
    # "fusion" -> trained scikit-learn model (needs a model file produced by
    #             backend/train_fusion_model.py to exist at fusion_model_path)
    consistency_backend: str = "rules"
    fusion_model_path: str = "app/ml/artifacts/fusion_model.joblib"

    # When true AND vision_backend=="gemini", pipeline.py also runs
    # TesseractOCR on the bill image and compares it against Gemini's
    # extraction (see services/consistency.py::compare_bill_extractions).
    # Off by default because with the default mock-only setup there's
    # only one extraction to begin with -- nothing to cross-check yet.
    enable_cross_check: bool = False

    # Redact flag explanations down to just the category name for
    # claimant-facing responses -- a detailed "bill exceeds reference by
    # 40%" explanation is useful to an adjuster and a cheat sheet for a
    # claimant deciding what to fix before resubmitting. See
    # api/claims.py's `audience` query param.
    expose_explanations_to_claimant: bool = False

    # --- misc ---
    upload_dir: str = "uploads"
    reused_photo_hamming_threshold: int = 6  # pHash distance considered "same photo"


settings = Settings()
