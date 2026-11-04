"""Shared Claude API settings and token accounting."""

from dataclasses import dataclass
from typing import Any

MODEL = "claude-opus-5-5"
# If the main model declines a request, the API re-runs it on a fallback model in the same call.
BETAS = ["server-side-fallback-2026-07-01"]

# USD per million tokens for MODEL. Used for budget guards and reports, so totals are estimates.
INPUT_PRICE = 4.00
OUTPUT_PRICE = 20.00
CACHE_WRITE_PRICE = 5.00
CACHE_READ_PRICE = 0.20


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def add(self, usage: Any) -> None:
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * INPUT_PRICE
            + self.output_tokens * OUTPUT_PRICE
            + self.cache_write_tokens * CACHE_WRITE_PRICE
            + self.cache_read_tokens * CACHE_READ_PRICE
        ) / 1_000_000


def create_message(client: Any, **kwargs: Any) -> Any:
    """Send one Messages API request with the project's model and fallback settings."""
    return client.beta.messages.create(model=MODEL, betas=BETAS, fallbacks="default", **kwargs)
