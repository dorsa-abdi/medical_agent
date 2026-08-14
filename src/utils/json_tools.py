import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character == "{":
            try:
                value, _ = decoder.raw_decode(cleaned[index:])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue
    raise ValueError("The model did not return a JSON object")


def parse_model_json(text: str, schema: type[T]) -> T:
    return schema.model_validate(extract_json_object(text))
