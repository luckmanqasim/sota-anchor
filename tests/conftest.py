"""Shared fixtures. Every test here is offline: no test may touch the network."""

from __future__ import annotations

import datetime as dt

import pytest

NOW = dt.datetime(2026, 9, 19, tzinfo=dt.timezone.utc)


def ts(year: int, month: int, day: int) -> int:
    """Unix timestamp, the form OpenRouter uses for `created`."""
    return int(dt.datetime(year, month, day, tzinfo=dt.timezone.utc).timestamp())


def or_model(
    model_id: str,
    created: int,
    *,
    output_modalities: list[str] | None = None,
    supported_parameters: list[str] | None = None,
    expiration_date: str | None = None,
    knowledge_cutoff: str | None = None,
    context_length: int | None = 200000,
    name: str | None = None,
    prompt_price: str = "0.000003",
    hugging_face_id: str | None = None,
) -> dict:
    """One entry shaped like a real OpenRouter /api/v1/models record."""
    return {
        "id": model_id,
        "canonical_slug": model_id,
        "name": name or model_id,
        "created": created,
        "context_length": context_length,
        "knowledge_cutoff": knowledge_cutoff,
        "expiration_date": expiration_date,
        "architecture": {
            "modality": "text->text",
            "input_modalities": ["text"],
            "output_modalities": output_modalities if output_modalities is not None else ["text"],
            "tokenizer": "Other",
        },
        "hugging_face_id": hugging_face_id,
        "pricing": {"prompt": prompt_price, "completion": "0.000015"},
        "supported_parameters": (
            supported_parameters if supported_parameters is not None else ["tools", "response_format"]
        ),
        "top_provider": {"context_length": context_length},
    }


@pytest.fixture
def raw_models() -> list[dict]:
    """A catalog covering every tiering case observed in the live OpenRouter data."""
    return [
        # --- anthropic: three live lineages plus a superseded one ---
        or_model("anthropic/claude-fable-5.1", ts(2026, 9, 1), prompt_price="0.00001"),
        or_model("anthropic/claude-fable-5.1:batch", ts(2026, 9, 1)),
        or_model("anthropic/claude-opus-5", ts(2026, 7, 24), prompt_price="0.000005"),
        or_model("anthropic/claude-opus-4.8", ts(2026, 5, 27)),
        or_model("anthropic/claude-sonnet-5", ts(2026, 6, 30), prompt_price="0.000002"),
        or_model("anthropic/claude-sonnet-4.5", ts(2025, 9, 29)),
        or_model("anthropic/claude-haiku-4.5", ts(2025, 10, 15), prompt_price="0.000001"),
        # --- google: supersession, an image model, and an audio model ---
        or_model("google/gemini-3.8-flash", ts(2026, 9, 2), knowledge_cutoff="2026-03-01",
            prompt_price="0.00000075"),
        or_model("google/gemini-2.5-flash", ts(2025, 6, 17)),
        or_model("google/gemini-3.1-pro-preview", ts(2026, 2, 19), prompt_price="0.000002"),
        or_model("google/gemini-2.5-pro", ts(2025, 6, 17)),
        or_model(
            "google/gemini-3-pro-image",
            ts(2026, 6, 18),
            output_modalities=["image", "text"],
        ),
        or_model(
            "google/lyria-3-clip-preview",
            ts(2026, 3, 30),
            output_modalities=["text", "audio"],
            supported_parameters=["max_tokens"],
        ),
        # A newer but much cheaper open-weights model: must not displace the
        # flagship, which is what recency-only ranking did to gemini Pro.
        or_model(
            "google/gemma-4-26b-a4b-it",
            ts(2026, 4, 3),
            prompt_price="0.00000009",
            hugging_face_id="google/gemma-4-26B-A4B-it",
        ),
        # --- openai: orphan lineages that only temporal decay can demote ---
        or_model("openai/gpt-6-astra", ts(2026, 9, 4), prompt_price="0.00001"),
        or_model("openai/gpt-5.5", ts(2026, 4, 24), prompt_price="0.000005"),
        or_model("openai/gpt-4o", ts(2024, 5, 13)),
        or_model("openai/gpt-4-turbo", ts(2024, 4, 9)),
        or_model(
            "openai/gpt-3.5-turbo-instruct",
            ts(2023, 8, 28),
            supported_parameters=["max_tokens"],
        ),
        # --- an explicitly retired model ---
        or_model("openai/gpt-5.4-mini", ts(2026, 3, 17), expiration_date="2026-08-01"),
        # --- a floating alias namespace ---
        or_model("~google/gemini-pro-latest", ts(2026, 4, 27)),
        # --- a provider that simply ships slowly ---
        or_model("slowcorp/steady-1", ts(2025, 2, 1)),
    ]
