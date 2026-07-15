from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def load_exact_adapter_weights(
    model: Any, artifact_path: str | Path, *, adapter_name: str = "default"
) -> str:
    """Load with PEFT's official helpers and prove every in-memory tensor is exact."""
    import torch
    from peft.utils.save_and_load import (
        get_peft_model_state_dict,
        load_peft_weights,
        set_peft_model_state_dict,
    )

    source = load_peft_weights(str(artifact_path), device="cpu")
    set_peft_model_state_dict(model, source, adapter_name=adapter_name)
    loaded = get_peft_model_state_dict(model, adapter_name=adapter_name)
    mismatches = [
        key
        for key, source_value in source.items()
        if key not in loaded
        or not torch.equal(
            loaded[key].detach().float().cpu(), source_value.detach().float().cpu()
        )
    ]
    extra = sorted(set(loaded) - set(source))
    if mismatches or extra:
        raise ValueError(
            "in-memory PEFT adapter differs from its Artifact: "
            f"mismatched={mismatches[:10]} extra={extra[:10]}"
        )
    digest = hashlib.sha256()
    for key in sorted(source):
        digest.update(key.encode())
        digest.update(source[key].detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
