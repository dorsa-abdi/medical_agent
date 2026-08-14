from __future__ import annotations

import json
from typing import Any

from config.settings import Settings


def load_pretrained(factory: Any, model_id: str, **kwargs: Any) -> Any:
    """Use an existing model cache before attempting any network request."""
    try:
        return factory.from_pretrained(model_id, local_files_only=True, **kwargs)
    except OSError:
        return factory.from_pretrained(model_id, **kwargs)


class LocalHuggingFaceLLM:
    """Lazy CPU-friendly wrapper around a small Hugging Face instruct model."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline: Any | None = None
        self._tokenizer: Any | None = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise RuntimeError("Run `pip install -r requirements.txt` to enable the local LLM.") from exc
        self._tokenizer = load_pretrained(AutoTokenizer, self.settings.model_id)
        model = load_pretrained(
            AutoModelForCausalLM,
            self.settings.model_id,
            torch_dtype=torch.float32 if self.settings.device == "cpu" else "auto",
            low_cpu_mem_usage=True,
        )
        self._pipeline = pipeline(
            "text-generation", model=model, tokenizer=self._tokenizer,
            device=-1 if self.settings.device == "cpu" else 0,
        )

    def generate(self, system: str, user: str) -> str:
        self._load()
        prompt = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.settings.max_new_tokens,
            "return_full_text": False,
            "do_sample": self.settings.temperature > 0,
            "repetition_penalty": 1.05,
        }
        if self.settings.temperature > 0:
            kwargs["temperature"] = self.settings.temperature
        return self._pipeline(prompt, **kwargs)[0]["generated_text"].strip()

    def generate_json(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        from src.utils.json_tools import extract_json_object
        serialized = json.dumps(payload, ensure_ascii=False)
        first = self.generate(
            system + "\nReturn exactly one valid JSON object. Do not use Markdown or commentary.",
            serialized,
        )
        try:
            return extract_json_object(first)
        except ValueError:
            repaired = self.generate(
                "Convert the supplied model output into one valid JSON object. Preserve only "
                "facts present in the input. Return JSON only.",
                json.dumps({"original_input": payload, "invalid_output": first}, ensure_ascii=False),
            )
            return extract_json_object(repaired)


def load_llm(settings: Settings) -> LocalHuggingFaceLLM:
    return LocalHuggingFaceLLM(settings)
