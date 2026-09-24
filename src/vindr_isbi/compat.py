"""Compatibility glue for MMDetection 3.3, Torch 2.1, and Transformers 4.56.

The project intentionally keeps the CUDA-tested Torch 2.1/MMCV 2.1 ABI.  The
first Transformers release with DINOv3 moved several helpers out of
``modeling_utils`` and expects the public Torch pytree registration name added
after Torch 2.1.  MMDetection 3.3 and MMPReTrain 1.2 still import the old
symbols while registering all model families, even for detectors that do not
use language.

This module installs narrow aliases only; it does not replace implementations
or patch package files in the Conda environment.
"""

from __future__ import annotations

import inspect
from importlib import metadata, util
from typing import Any

from packaging.version import Version


def _install_torch_pytree_alias(pytree: Any) -> bool:
    """Expose Torch's public pytree API name when Torch 2.1 has only the private one."""

    if hasattr(pytree, "register_pytree_node"):
        return False
    private_register = getattr(pytree, "_register_pytree_node", None)
    if private_register is None:
        return False
    supported = set(inspect.signature(private_register).parameters)

    def register_pytree_node(
        typ: Any,
        flatten_fn: Any,
        unflatten_fn: Any,
        *,
        serialized_type_name: str | None = None,
        to_dumpable_context: Any = None,
        from_dumpable_context: Any = None,
    ) -> Any:
        # ``serialized_type_name`` was added with the public API and has no
        # Torch 2.1 equivalent.  It affects serialization metadata, not model
        # execution, so it is deliberately ignored on that version.
        del serialized_type_name
        kwargs = {}
        if "to_dumpable_context" in supported:
            kwargs["to_dumpable_context"] = to_dumpable_context
        if "from_dumpable_context" in supported:
            kwargs["from_dumpable_context"] = from_dumpable_context
        return private_register(typ, flatten_fn, unflatten_fn, **kwargs)

    pytree.register_pytree_node = register_pytree_node
    return True


def _install_transformers_chunking_alias(modeling_utils: Any, pytorch_utils: Any) -> bool:
    """Restore the Transformers symbol imported by MMDetection 3.3's GLIP code."""

    if hasattr(modeling_utils, "apply_chunking_to_forward"):
        return False
    chunking = getattr(pytorch_utils, "apply_chunking_to_forward", None)
    if chunking is None:
        return False
    modeling_utils.apply_chunking_to_forward = chunking
    return True


def _install_transformers_pruning_aliases(
    modeling_utils: Any,
    pytorch_utils: Any,
) -> tuple[str, ...]:
    """Restore helpers imported by MMPReTrain 1.2's legacy BLIP modules."""

    installed = []
    for name in ("find_pruneable_heads_and_indices", "prune_linear_layer"):
        if hasattr(modeling_utils, name):
            continue
        implementation = getattr(pytorch_utils, name, None)
        if implementation is None:
            continue
        setattr(modeling_utils, name, implementation)
        installed.append(name)
    return tuple(installed)


def _install_transformers_generation_aliases(
    modeling_utils: Any,
    generation_utils: Any,
) -> tuple[str, ...]:
    """Restore generation classes imported by MMPReTrain 1.2's OFA code."""

    installed = []
    for name in ("GenerationConfig", "GenerationMixin"):
        if hasattr(modeling_utils, name):
            continue
        implementation = getattr(generation_utils, name, None)
        if implementation is None:
            continue
        setattr(modeling_utils, name, implementation)
        installed.append(name)
    return tuple(installed)


def ensure_mmdet_transformers_compat(*, strict: bool = True) -> dict[str, Any]:
    """Prepare optional Transformers before MMEngine/MMDetection imports.

    Returns a small diagnostic dictionary.  When Transformers is absent the
    function is a no-op because ordinary MMDetection models do not require it.
    Transformers 5 is rejected explicitly: it requires a newer Torch ABI than
    the project's compiled MMCV stack.
    """

    result: dict[str, Any] = {
        "transformers_installed": False,
        "pytree_alias_installed": False,
        "chunking_alias_installed": False,
        "pruning_aliases_installed": (),
        "generation_aliases_installed": (),
    }
    if util.find_spec("transformers") is None:
        return result

    transformers_version = Version(metadata.version("transformers"))
    result.update(
        transformers_installed=True,
        transformers_version=str(transformers_version),
    )
    if transformers_version.major >= 5:
        message = (
            "Transformers 5 is incompatible with the project's Torch 2.1/MMCV 2.1 "
            "runtime. Install `transformers==4.56.2` and `huggingface-hub>=0.34,<1`."
        )
        if strict:
            raise RuntimeError(message)
        result["error"] = message
        return result

    try:
        import torch.utils._pytree as torch_pytree

        result["pytree_alias_installed"] = _install_torch_pytree_alias(torch_pytree)

        import transformers.generation as generation_utils
        import transformers.modeling_utils as modeling_utils
        import transformers.pytorch_utils as pytorch_utils

        result["chunking_alias_installed"] = _install_transformers_chunking_alias(
            modeling_utils,
            pytorch_utils,
        )
        result["pruning_aliases_installed"] = _install_transformers_pruning_aliases(
            modeling_utils,
            pytorch_utils,
        )
        result["generation_aliases_installed"] = _install_transformers_generation_aliases(
            modeling_utils,
            generation_utils,
        )
    except Exception as exc:
        if strict:
            raise RuntimeError(
                "Could not initialize the MMDetection/Transformers compatibility layer. "
                "Verify torch==2.1.*, transformers==4.56.2, and "
                "huggingface-hub>=0.34,<1."
            ) from exc
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


__all__ = [
    "ensure_mmdet_transformers_compat",
]
