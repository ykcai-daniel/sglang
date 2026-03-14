import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from safetensors import safe_open

from sglang.multimodal_gen.runtime.layers.quantization import (
    QuantizationConfig,
    get_quantization_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def find_quant_modelslim_config(model_config, component_model_path):
    quant_config_file = Path(component_model_path, "quant_model_description.json")
    quant_cfg = None
    if quant_config_file.is_file():
        with open(quant_config_file) as f:
            quant_cfg = json.load(f)
        # This field is required for flagless model loading but is not present in
        # modelslim model description, so we're adding it here manually.
        quant_cfg["quant_method"] = "modelslim"

    return quant_cfg


def replace_prefix(key: str, prefix_mapping: dict[str, str]) -> str:
    for prefix, new_prefix in prefix_mapping.items():
        if key.startswith(prefix):
            key = key.replace(prefix, new_prefix, 1)
    return key


def get_quant_config(
    model_config,
    component_model_path: str,
    packed_modules_mapping: Dict[str, List[str]] = {},
    remap_prefix: Dict[str, str] | None = None,
) -> QuantizationConfig:

    quant_cfg = find_quant_modelslim_config(model_config, component_model_path)
    if quant_cfg is not None:
        quant_cls = get_quantization_config(quant_cfg["quant_method"])
        return quant_cls.from_config(quant_cfg)
    else:
        if "quantization_config" not in model_config:
            return None
        quant_cls = get_quantization_config(
            model_config["quantization_config"]["quant_method"]
        )

        # GGUF doesn't have config file
        if model_config["quantization_config"]["quant_method"] == "gguf":
            return quant_cls.from_config({})

        # Read the quantization config from the HF model config, if available.
        hf_quant_config = model_config["quantization_config"]
        # some vision model may keep quantization_config in their text_config
        hf_text_config = getattr(model_config, "text_config", None)
        if hf_quant_config is None and hf_text_config is not None:
            hf_quant_config = getattr(hf_text_config, "quantization_config", None)
        if hf_quant_config is None:
            # compressed-tensors uses a compressions_config
            hf_quant_config = getattr(model_config, "compression_config", None)
        if hf_quant_config is not None:
            hf_quant_config["packed_modules_mapping"] = packed_modules_mapping
            return quant_cls.from_config(hf_quant_config)
        # In case of bitsandbytes/QLoRA, get quant config from the adapter model.
        else:
            model_name_or_path = model_config["model_path"]
        is_local = os.path.isdir(model_name_or_path)
        hf_folder = model_name_or_path

        possible_config_filenames = quant_cls.get_config_filenames()

        # If the quantization config is not found, use the default config.
        if not possible_config_filenames:
            return quant_cls()

        config_files = glob.glob(os.path.join(hf_folder, "*.json"))

        quant_config_files = [
            f
            for f in config_files
            if any(f.endswith(x) for x in possible_config_filenames)
        ]
        if len(quant_config_files) == 0:
            raise ValueError(
                f"Cannot find the config file for {model_config['quantization_config']['quant_method']}"
            )
        if len(quant_config_files) > 1:
            raise ValueError(
                f"Found multiple config files for {model_config['quantization_config']['quant_method']}: "
                f"{quant_config_files}"
            )

        quant_config_file = quant_config_files[0]
        with open(quant_config_file) as f:
            config = json.load(f)
            if remap_prefix is not None:
                exclude_modules = [
                    replace_prefix(key, remap_prefix)
                    for key in config["quantization"]["exclude_modules"]
                ]
                config["quantization"]["exclude_modules"] = exclude_modules
            config["packed_modules_mapping"] = packed_modules_mapping
            return quant_cls.from_config(config)


def handle_fp8_metadata_format(quant_config_dict):
    layers = quant_config_dict.get("layers", {})
    if any(
        isinstance(v, dict) and "float8" in v.get("format", "") for v in layers.values()
    ):
        quant_config_dict["quant_method"] = "fp8"
        quant_config_dict["activation_scheme"] = "dynamic"
    return quant_config_dict


def get_quant_config_from_safetensors_metadata(
    file_path: str,
) -> Optional[QuantizationConfig]:
    """Extract quantization config from a safetensors file's metadata header.
    Returns None if no recognizable quantization metadata is found.
    """
    metadata = get_metadata_from_safetensors_file(file_path)
    if not metadata:
        return None

    quant_config_str = metadata.get("_quantization_metadata")
    if not quant_config_str:
        return None
    try:
        quant_config_dict = json.loads(quant_config_str)
    except Exception as _e:
        return None

    # handle diffusers fp8 safetensors metadata format
    if (
        "quant_method" not in quant_config_dict
        and "format_version" in quant_config_dict
        and "layers" in quant_config_dict
    ):
        quant_config_dict = handle_fp8_metadata_format(quant_config_dict)

    quant_method = quant_config_dict.get("quant_method")
    if not quant_method:
        return None

    try:
        quant_cls = get_quantization_config(quant_method)
        config = quant_cls.from_config(quant_config_dict)
        logger.debug(f"Get quantization config from safetensors file: {file_path}")
        return config
    except Exception as _e:
        return None


def get_metadata_from_safetensors_file(file_path: str):
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            return metadata
    except Exception as e:
        logger.warning(e)


def build_nvfp4_config_from_safetensors(
    file_path: str,
    param_names_mapping_dict: Optional[dict] = None,
) -> Optional[QuantizationConfig]:
    """Build ModelOptFp4Config from safetensors with NVFP4 per-layer format metadata.

    Handles the case where _quantization_metadata uses:
        {"format_version": "1.0", "layers": {"layer_name": {"format": "nvfp4"}, ...}}
    instead of the standard {"quant_method": "modelopt_fp4", ...} format.

    Args:
        file_path: Path to the safetensors file.
        param_names_mapping_dict: Optional dict mapping BFL param name patterns to HF
            param name patterns (from DiTArchConfig.param_names_mapping). Used to
            convert non-quantized layer names to HF format for exclude_modules.
    """
    metadata = get_metadata_from_safetensors_file(file_path)
    if not metadata:
        return None

    quant_config_str = metadata.get("_quantization_metadata")
    if not quant_config_str:
        return None

    quant_config_dict = json.loads(quant_config_str)

    # Only handle the per-layer format (format_version + layers)
    if "format_version" not in quant_config_dict or "layers" not in quant_config_dict:
        return None

    layers = quant_config_dict.get("layers", {})
    formats = {v.get("format", "") for v in layers.values() if isinstance(v, dict)}
    if "nvfp4" not in formats:
        return None

    # Infer group_size from tensor shapes and collect non-quantized weight tensors

    group_size = None
    non_quantized_bfl_modules = []
    import torch
    with safe_open(file_path, framework="pt", device="cpu") as f:
        all_keys = set(f.keys())
        # Infer group_size from first quantized layer's weight and weight_scale shapes
        for layer_name in layers:
            weight_key = f"{layer_name}.weight"
            scale_key = f"{layer_name}.weight_scale"
            if weight_key in all_keys and scale_key in all_keys:
                w = f.get_tensor(weight_key)
                s = f.get_tensor(scale_key)
                # FP4 weight: [out, in//2] uint8; scale: [out, in//group_size] fp8
                input_size = w.shape[1] * 2
                group_size = input_size // s.shape[1]
                break

        # Collect bf16 weight tensors — these are the non-quantized layers
        for k in sorted(all_keys):
            if k.endswith(".weight"):
                t = f.get_tensor(k)
                if t.dtype != torch.uint8:
                    non_quantized_bfl_modules.append(k[: -len(".weight")])

    if group_size is None:
        logger.warning("Could not infer group_size from NVFP4 safetensors: %s", file_path)
        return None

    # Convert BFL module names to HF names using param_names_mapping
    exclude_modules = []
    if param_names_mapping_dict:
        from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping

        mapping_fn = get_param_names_mapping(param_names_mapping_dict)
        for module_bfl in non_quantized_bfl_modules:
            # Append .weight so the mapping patterns (which match full param names) work
            mapped, _, _ = mapping_fn(f"{module_bfl}.weight")
            exclude_modules.append(mapped[: -len(".weight")] if mapped.endswith(".weight") else mapped)
    else:
        exclude_modules = non_quantized_bfl_modules

    try:
        quant_cls = get_quantization_config("modelopt_fp4")
        result = quant_cls.from_config(
            {"quant_algo": "NVFP4", "group_size": group_size, "ignore": exclude_modules}
        )
        logger.info(
            "Built NVFP4 quant config from %s: group_size=%d, %d excluded modules",
            file_path,
            group_size,
            len(exclude_modules),
        )
        return result
    except Exception as e:
        logger.warning("Failed to build NVFP4 config from %s: %s", file_path, e)
        return None
