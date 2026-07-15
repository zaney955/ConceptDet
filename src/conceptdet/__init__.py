"""Qwen3-VL reference-guided Detection Sets without segmentation dependencies."""

from conceptdet.application import DetectionApplication, DetectionResult
from conceptdet.config import RequestConfig
from conceptdet.types import Box

__all__ = ["Box", "DetectionApplication", "DetectionResult", "RequestConfig"]
__version__ = "0.7.0"
