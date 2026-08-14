import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def _dotenv_values(path: Path = Path(".env")) -> dict[str, str]:
    if not path.exists():
        return {}
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class Settings(BaseModel):
    """Runtime settings with lightweight environment-variable loading."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    model_id: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct", alias="MODEL_ID")
    device: str = Field(default="cpu", alias="DEVICE")
    max_new_tokens: int = Field(default=384, alias="MAX_NEW_TOKENS")
    temperature: float = Field(default=0.0, alias="TEMPERATURE")
    max_follow_up_turns: int = Field(default=12, alias="MAX_FOLLOW_UP_TURNS")
    question_timeout_seconds: float = Field(default=90.0, gt=0, alias="QUESTION_TIMEOUT_SECONDS")
    emergency_number: str = Field(default="your local emergency number", alias="EMERGENCY_NUMBER")
    knowledge_dir: Path = Field(default=Path("data/knowledge"), alias="KNOWLEDGE_DIR")
    top_k: int = Field(default=3, ge=1, le=10, alias="RAG_TOP_K")
    use_llm_reasoning: bool = Field(default=True, alias="USE_LLM_REASONING")
    delphi_checkpoint: Path | None = Field(default=None, alias="DELPHI_CHECKPOINT")
    delphi_vocabulary: Path = Field(default=Path("data/delphi/vocabulary.json"), alias="DELPHI_VOCABULARY")
    delphi_horizon_years: float = Field(default=5.0, gt=0, alias="DELPHI_HORIZON_YEARS")

    def __init__(self, **values):
        dotenv = _dotenv_values()
        environment = {}
        for name, field in type(self).model_fields.items():
            alias = field.alias
            if not alias or name in values or alias in values:
                continue
            if alias in os.environ:
                environment[name] = os.environ[alias]
            elif alias in dotenv:
                environment[name] = dotenv[alias]
        super().__init__(**environment, **values)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
