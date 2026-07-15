class ConceptDetError(Exception):
    """Base error for expected ConceptDet failures."""


class InputError(ConceptDetError):
    """Raised when a request contains invalid input."""


class ModelLoadError(ConceptDetError):
    """Raised when a checkpoint is incompatible with the detection backend."""


class OutputFormatError(ConceptDetError):
    """Raised when model output does not contain a usable bounding box."""
