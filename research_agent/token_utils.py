"""Token estimation helpers (tiktoken when possible, char heuristic fallback)."""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=8)
def _encoder_for(model_hint: str):
    try:
        import tiktoken

        try:
            return tiktoken.encoding_for_model(model_hint)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def estimate_tokens(text: str, model_hint: str = "gpt-4o-mini") -> int:
    enc = _encoder_for(model_hint)
    if enc is None:
        return max(1, len(text) // 4)
    return max(1, len(enc.encode(text)))
