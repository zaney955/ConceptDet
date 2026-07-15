from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from conceptdet.types import Box


@dataclass(frozen=True)
class AdapterInput:
    reference_image: Image.Image
    reference_boxes: tuple[Box, ...]
    target_image: Image.Image
    query: str


@dataclass(frozen=True)
class AdapterGeneration:
    completion: str
    prepared_reference: Image.Image
    prepared_target: Image.Image
    image_grids: tuple[tuple[int, int, int], ...] = ()
    prompt_tokens: int | None = None


class DetectionAdapter(Protocol):
    def generate(self, request: AdapterInput, *, max_new_tokens: int) -> AdapterGeneration: ...


class FakeAdapter:
    """Deterministic Adapter used through the production application seam."""

    def __init__(self, completion: str = "[]") -> None:
        self.completion = completion
        self.requests: list[AdapterInput] = []

    def generate(self, request: AdapterInput, *, max_new_tokens: int) -> AdapterGeneration:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.requests.append(request)
        return AdapterGeneration(
            self.completion,
            request.reference_image.copy(),
            request.target_image.copy(),
        )
