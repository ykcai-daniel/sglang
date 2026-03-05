# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""Utilities for selecting and loading models."""

import contextlib
import glob
import os
import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any, Dict, Type

import torch
from torch import nn

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def get_param_names_mapping(
    mapping_dict: dict[str, str | tuple[str, int, int] | list[str]],
) -> Callable[[str], tuple[str | list[str], Any, Any]]:
    """
    Creates a mapping function that transforms parameter names using regex patterns.

    Args:
        mapping_dict: Dictionary mapping regex patterns to replacement patterns.
            Values can be:
            - str: 1-to-1 rename
            - tuple(str, int, int): N-to-1 merge (target, merge_index, total)
            - list[str]: 1-to-N split (each element is a separate target name pattern)

    Returns:
        A function that maps parameter names from source to target format.
        Returns (target_name, merge_index, total_split_params) where target_name
        may be a list[str] when a split mapping was applied.
    """

    def mapping_fn(name: str) -> tuple[str | list[str], Any, Any]:
        # support chained conversions, e.g.:
        # transformer.xxx.lora_down -> xxx.lora_down -> xxx.proj_down
        merge_index = None
        total_split_params = None
        max_steps = max(8, len(mapping_dict) * 2)
        applied_patterns: set[str] = set()
        visited_names: set[str] = {name}

        for _ in range(max_steps):
            transformed = False
            for pattern, replacement in mapping_dict.items():
                # avoid re-applying the same rule on its own output
                if pattern in applied_patterns:
                    continue
                if re.match(pattern, name) is None:
                    continue

                curr_merge_index = None
                curr_total_split_params = None

                # 1-to-N split: replacement is a list of target patterns
                if isinstance(replacement, list):
                    new_names = [re.sub(pattern, r, name) for r in replacement]
                    # Only apply if all new names differ from the source
                    if all(n != name for n in new_names):
                        applied_patterns.add(pattern)
                        return new_names, None, None
                    continue

                if isinstance(replacement, tuple):
                    curr_merge_index = replacement[1]
                    curr_total_split_params = replacement[2]
                    replacement = replacement[0]

                new_name = re.sub(pattern, replacement, name)

                if new_name != name:
                    if curr_merge_index is not None:
                        merge_index = curr_merge_index
                        total_split_params = curr_total_split_params

                    name = new_name
                    applied_patterns.add(pattern)
                    if name in visited_names:
                        transformed = False
                        break
                    visited_names.add(name)
                    transformed = True
                    break

            if not transformed:
                break

        return name, merge_index, total_split_params

    return mapping_fn


def hf_to_custom_state_dict(
    hf_param_sd: dict[str, torch.Tensor] | Iterator[tuple[str, torch.Tensor]],
    param_names_mapping: Callable[[str], tuple[str, Any, Any]],
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[str, Any, Any]]]:
    """
    Converts a Hugging Face parameter state dictionary to a custom parameter state dictionary.

    Args:
        hf_param_sd (Dict[str, torch.Tensor]): The Hugging Face parameter state dictionary
        param_names_mapping (Callable[[str], tuple[str, Any, Any]]): A function that maps parameter names from source to target format

    Returns:
        custom_param_sd (Dict[str, torch.Tensor]): The custom formatted parameter state dict
        reverse_param_names_mapping (Dict[str, Tuple[str, Any, Any]]): Maps back from custom to hf
    """
    custom_param_sd = {}
    to_merge_params = defaultdict(dict)  # type: ignore
    reverse_param_names_mapping = {}
    if isinstance(hf_param_sd, dict):
        hf_param_sd = hf_param_sd.items()  # type: ignore
    for source_param_name, full_tensor in hf_param_sd:  # type: ignore
        target_param_name, merge_index, num_params_to_merge = param_names_mapping(
            source_param_name
        )
        if target_param_name == "" or target_param_name is None:  # type: ignore[comparison-overlap]
            continue

        # 1-to-N split: target_param_name is a list of target names
        if isinstance(target_param_name, list):
            n = len(target_param_name)
            if full_tensor.numel() > 1 and full_tensor.shape[0] % n == 0:
                # Split structured tensor along dim 0 (output dimension)
                slice_size = full_tensor.shape[0] // n
                for i, tname in enumerate(target_param_name):
                    custom_param_sd[tname] = full_tensor.narrow(0, i * slice_size, slice_size).clone()
                    reverse_param_names_mapping[tname] = (source_param_name, None, None)
            else:
                # Scalar or indivisible tensor: broadcast to all targets
                for tname in target_param_name:
                    custom_param_sd[tname] = full_tensor.clone()
                    reverse_param_names_mapping[tname] = (source_param_name, None, None)
            continue

        reverse_param_names_mapping[target_param_name] = (
            source_param_name,
            merge_index,
            num_params_to_merge,
        )
        if merge_index is not None:
            to_merge_params[target_param_name][merge_index] = full_tensor
            if len(to_merge_params[target_param_name]) == num_params_to_merge:
                # cat at output dim according to the merge_index order
                sorted_tensors = [
                    to_merge_params[target_param_name][i]
                    for i in range(num_params_to_merge)
                ]
                full_tensor = torch.cat(sorted_tensors, dim=0)
                del to_merge_params[target_param_name]
            else:
                continue
        custom_param_sd[target_param_name] = full_tensor
    return custom_param_sd, reverse_param_names_mapping


class skip_init_modules:
    def __enter__(self):
        # Save originals
        self._orig_reset = {}
        for cls in (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d):
            self._orig_reset[cls] = cls.reset_parameters
            cls.reset_parameters = lambda self: None  # skip init

    def __exit__(self, exc_type, exc_value, traceback):
        # restore originals
        for cls, orig in self._orig_reset.items():
            cls.reset_parameters = orig


def _normalize_component_type(module_type: str) -> str:
    """Normalize module types like 'text_encoder_2' -> 'text_encoder'."""
    if module_type.endswith("_2"):
        return module_type[:-2]
    return module_type


def _clean_hf_config_inplace(model_config: dict) -> None:
    """Remove common extraneous HF fields if present."""
    for key in (
        "_name_or_path",
        "transformers_version",
        "model_type",
        "tokenizer_class",
        "torch_dtype",
    ):
        model_config.pop(key, None)


def _list_safetensors_files(model_path: str) -> list[str]:
    """List all .safetensors files under a directory."""
    return sorted(glob.glob(os.path.join(str(model_path), "*.safetensors")))


BYTES_PER_GB = 1024**3


def get_memory_usage_of_component(module) -> float | None:
    """
    returned value is in GB, rounded to 2 decimal digits
    """
    if not isinstance(module, nn.Module):
        return None
    if hasattr(module, "get_memory_footprint"):
        usage = module.get_memory_footprint() / BYTES_PER_GB
    else:
        # manually
        param_size = sum(p.numel() * p.element_size() for p in module.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in module.buffers())

        total_size_bytes = param_size + buffer_size
        usage = total_size_bytes / (1024**3)

    return round(usage, 2)


# component name ->  ComponentLoader class
component_name_to_loader_cls: Dict[str, Type[Any]] = {}
