from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def _model_id(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        value = item.get("id") or item.get("name") or item.get("model")
        return str(value) if value else None
    return None


def list_openai_compatible_models(
    *,
    base_url: str,
    api_key_env: str,
    expect: list[str] | None = None,
    contains: str | None = None,
    request_timeout: float = 30.0,
) -> dict[str, Any]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} is not set")

    endpoint = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"model list failed with HTTP {exc.code}: {detail}") from exc

    raw_models = payload.get("data", payload.get("models", payload if isinstance(payload, list) else []))
    model_ids = sorted(model_id for item in raw_models if (model_id := _model_id(item)))
    if contains:
        model_ids = [model_id for model_id in model_ids if contains.lower() in model_id.lower()]

    expected = expect or []
    return {
        "base_url": base_url,
        "endpoint": endpoint,
        "api_key_env": api_key_env,
        "models": model_ids,
        "count": len(model_ids),
        "expected": {model: model in model_ids for model in expected},
        "raw_response_type": type(payload).__name__,
    }
