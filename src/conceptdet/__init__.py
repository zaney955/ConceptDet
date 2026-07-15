"""Reference-guided concept detection without segmentation dependencies."""

from conceptdet.pipeline import DetectionPipeline, DetectionRequest, DetectionResult
from conceptdet.types import Box

__all__ = ["Box", "DetectionPipeline", "DetectionRequest", "DetectionResult"]
__version__ = "0.1.0"
