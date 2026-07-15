class ConceptDetError(Exception):
    """Base error for expected ConceptDet failures."""


class InputError(ConceptDetError):
    """Raised when a request contains invalid input."""


class ModelLoadError(ConceptDetError):
    """Raised when a checkpoint is incompatible with the detection backend."""


class OutputFormatError(ConceptDetError):
    """Raised when model output does not contain a usable bounding box."""


class ConfigurationError(ConceptDetError):
    """Raised when a versioned ConceptDet YAML document is invalid."""


class ArtifactError(ConceptDetError):
    """Raised when an Adapter Artifact is missing, mutable, or incompatible."""


class DatasetError(ConceptDetError):
    """Raised when a compiled Detection Set dataset is invalid or incompatible."""


class TrainingError(ConceptDetError):
    """Raised when an SFT stage violates its data, lifecycle, or resource contract."""


class EvaluationError(ConceptDetError):
    """Raised when evaluation inputs or frozen report publication are invalid."""
