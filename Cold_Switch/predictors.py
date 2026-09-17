"""Replaceable output-length predictors; importing the Router does not load torch."""

from __future__ import annotations

import importlib
import json
import math
import numbers
import sys
from pathlib import Path
from typing import Protocol


class LengthPredictor(Protocol):
    def predict(self, prompt_text: str) -> int:
        """Return the estimated number of output tokens."""
        ...


def validate_prediction(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("predict() must return a finite non-negative token count")
    if not math.isfinite(value) or value < 0:
        raise ValueError("predict() must return a finite non-negative token count")
    return int(round(value))


def request_type_for(
    input_tokens: int, output_tokens: int, input_threshold: int, output_threshold: int
) -> str:
    prompt = "long" if input_tokens >= input_threshold else "short"
    response = "long" if output_tokens >= output_threshold else "short"
    return f"{prompt}_prompt_{response}_response"


def bert_predictor(
    model_dir: str | None = None, device: str = "cpu"
) -> LengthPredictor:
    root = Path(__file__).resolve().parent.parent
    directory = (
        Path(model_dir).expanduser()
        if model_dir
        else root / "Bert_Classifier" / "bert_llama3_70B"
    )
    if not (directory / "best_model.pt").is_file():
        raise FileNotFoundError(
            f"Missing {directory / 'best_model.pt'}. Download best_model.pt from the "
            "repository Releases and place it beside the tokenizer files in "
            "Bert_Classifier/bert_llama3_70B/, or set model_dir in --predictor-kwargs."
        )
    # Permit `python router.py` from Cold_Switch without installing the repository.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from Bert_Classifier.predict import TokenCountPredictor

    return TokenCountPredictor(model_dir=str(directory), device=device)


def load_predictor(factory_spec: str, kwargs_json: str) -> LengthPredictor:
    module_name, separator, attribute = factory_spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("predictor factory must be module:callable")
    kwargs = json.loads(kwargs_json)
    if not isinstance(kwargs, dict):
        raise ValueError("predictor kwargs must be a JSON object")
    factory = getattr(importlib.import_module(module_name), attribute)
    predictor = factory(**kwargs)
    if not callable(getattr(predictor, "predict", None)):
        raise TypeError("predictor must expose predict(prompt_text) -> token_count")
    return predictor
