# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/modelopt_quant.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.models.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.layers.utils.common import copy_or_rebind_param
from sglang.srt.utils.custom_op import register_custom_op

# FP4 activation quantization kernel (flashinfer on Blackwell, sgl_kernel elsewhere).
fp4_quantize = None
try:
    if current_platform.is_sm120() or current_platform.is_blackwell():
        from flashinfer import fp4_quantize
    else:
        from sgl_kernel import scaled_fp4_quant as fp4_quantize
except ImportError:
    pass

# comfy_kitchen cuBLAS NVFP4 backend (optional, Blackwell-only).
ck_cuda = None
_ck_available = False
try:
    import comfy_kitchen.backends.cuda as ck_cuda

    _ck_available = True
except Exception:
    pass

# FP4 GEMM: prefer flashinfer, fall back to sgl_kernel CUTLASS.
flashinfer_fp4_gemm = None
enable_flashinfer_fp4_gemm = False
cutlass_fp4_gemm = None
try:
    from flashinfer import mm_fp4 as flashinfer_fp4_gemm

    enable_flashinfer_fp4_gemm = True
except ImportError:
    if current_platform.is_cuda():
        from sgl_kernel import cutlass_scaled_fp4_mm as cutlass_fp4_gemm


# Initialize logger for the module
logger = logging.getLogger(__name__)


def _sglang_fp4_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
) -> torch.Tensor:
    M = input.shape[-2]
    N = int(out_features)
    return input.new_empty((M, N), dtype=out_dtype)


@register_custom_op(fake_impl=_sglang_fp4_gemm_fake)
def fp4_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
) -> torch.Tensor:
    backend = FLASHINFER_FP4_GEMM_BACKEND if FLASHINFER_FP4_GEMM_BACKEND else "cudnn"
    if enable_flashinfer_fp4_gemm:
        return flashinfer_fp4_gemm(
            input,
            weight,
            input_sf,
            weight_sf,
            alpha,
            out_dtype,
            backend=backend,
            skip_check=False,
        )
    else:
        if cutlass_fp4_gemm is None:
            raise RuntimeError(
                "Neither flashinfer nor sgl_kernel.cutlass_scaled_fp4_mm is available. "
                "Install flashinfer or upgrade sgl_kernel to use fp4_gemm."
            )
        return cutlass_fp4_gemm(input, weight, input_sf, weight_sf, alpha, out_dtype)


# Comment out as it can't re-register the same sgl_kernel::scaled_fp4_quant
# if (
#     current_platform.is_cuda()
#     and (not current_platform.is_sm120())
#     and (fp4_quantize is not None)
# ):

#     @register_fake_if_exists("sgl_kernel::scaled_fp4_quant")
#     def _sgl_kernel_scaled_fp4_quant_fake(
#         output, input, output_scale, input_global_scale
#     ):
#         return


FLASHINFER_FP4_GEMM_BACKEND = None

FP4_GEMM_ALIGNMENT = 32


def round_up_to_multiple(x: int, m: int) -> int:
    """Round up x to the nearest multiple of m."""
    return (x + m - 1) // m * m


def pad_nvfp4_weight(
    weight: torch.Tensor,
    n_alignment: int = FP4_GEMM_ALIGNMENT,
    k_alignment: int = FP4_GEMM_ALIGNMENT,
) -> tuple:
    """
    Pad packed NVFP4 weights to satisfy alignment constraints for FP4 GEMM kernels.

    Different backends have different alignment requirements:
    - CUTLASS/cuDNN: N % 32 == 0, K % 32 == 0
    - TRTLLM: N % 128 == 0 (for shuffle_matrix_sf_a), K padding handled separately

    Args:
        weight: Packed FP4 weight tensor of shape [N, K//2] (2 FP4 values per byte)
        n_alignment: Required alignment for N dimension (default 32, use 128 for TRTLLM)
        k_alignment: Required alignment for K dimension (default 32, use 0 to skip)

    Returns:
        Tuple of (padded_weight, weights_padding_cols) where weights_padding_cols
        is the number of columns added for K-dimension padding (in bytes).
    """
    weight_current_rows = weight.shape[0]  # N dimension
    weight_current_col_bytes = weight.shape[1]  # K//2 (packed)

    # Calculate padding for N dimension (rows)
    pad_rows = 0
    if n_alignment > 0 and weight_current_rows % n_alignment != 0:
        total_rows = round_up_to_multiple(weight_current_rows, n_alignment)
        pad_rows = total_rows - weight_current_rows

    # Calculate padding for K dimension (columns)
    # 2 FP4 items are packed per byte in the input dimension
    weight_current_col_elements = weight_current_col_bytes * 2
    pad_cols_bytes = 0
    if k_alignment > 0 and weight_current_col_elements % k_alignment != 0:
        total_cols = round_up_to_multiple(weight_current_col_elements, k_alignment)
        pad_cols = total_cols - weight_current_col_elements
        # pad_cols is in elements, but padding is in bytes (2 elements per byte)
        pad_cols_bytes = pad_cols // 2

    # Apply padding in a single operation if needed
    if pad_rows > 0 or pad_cols_bytes > 0:
        weight = torch.nn.functional.pad(
            weight, (0, pad_cols_bytes, 0, pad_rows)
        ).contiguous()

    return weight, pad_cols_bytes


def pad_nvfp4_activation_for_cutlass(
    x_fp4: torch.Tensor,
    weights_padding_cols: int,
) -> torch.Tensor:
    """Pad packed FP4 activations to match the K-dimension padding applied to weights."""
    if weights_padding_cols > 0:
        return torch.nn.functional.pad(x_fp4, (0, weights_padding_cols)).contiguous()
    return x_fp4


def slice_nvfp4_output(
    out: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """Slice the output tensor to remove N-dimension padding added to weights."""
    if out.shape[-1] != output_size:
        return out[..., :output_size].contiguous()
    return out


# Supported activation schemes for the current configuration
ACTIVATION_SCHEMES = ["static"]


class ModelOptQuantConfig(QuantizationConfig):
    def __init__(
        self,
        exclude_modules: Optional[List[str]],
        packed_modules_mapping: Optional[Dict[str, List[str]]],
    ):
        super().__init__()
        self.packed_modules_mapping = packed_modules_mapping or {}
        self.exclude_modules = exclude_modules or []

    def _get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        *,
        Linear: type[LinearMethodBase],
    ) -> Optional[QuantizeMethodBase]:
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            if self.is_layer_excluded(prefix) or (
                self.packed_modules_mapping
                and is_layer_skipped(prefix, [], self.packed_modules_mapping)
            ):
                return UnquantizedLinearMethod()
            return Linear(self)
        return None

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["hf_quant_config.json"]

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def override_quantization_method(cls, hf_quant_config, user_quant) -> Optional[str]:
        """Shared ModelOpt quantization method override logic."""
        if hf_quant_config is None:
            return None

        # Check if this is a ModelOpt config
        quant_algo = hf_quant_config.get("quant_algo", "").upper()

        # If user specified generic "modelopt", auto-detect the specific method
        if user_quant == "modelopt":
            if not ("NVFP4" in quant_algo or "FP4" in quant_algo):
                logger.warning(
                    f"Unsupported quant_algo '{quant_algo}' for user_quant 'modelopt'. Using the default 'modelopt_fp4' quant_algo."
                )

            # The hf_quant_config may be a parsed quant config, so we need to check the quant_method.
            return "modelopt_fp4"

        return None


class ModelOptFp4Config(ModelOptQuantConfig):
    """Config class for NVFP4."""

    def __init__(
        self,
        is_checkpoint_nvfp4_serialized: bool = False,
        group_size: int = None,
        exclude_modules: List[str] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        super().__init__(exclude_modules, packed_modules_mapping)
        self.is_checkpoint_nvfp4_serialized = is_checkpoint_nvfp4_serialized
        if is_checkpoint_nvfp4_serialized:
            logger.warning(
                "Detected nvfp4 checkpoint. Please note that the "
                "format is experimental and subject to change."
            )
        self.group_size = group_size

    @classmethod
    def get_name(cls) -> str:
        return "modelopt_fp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @staticmethod
    def common_group_size(cfg: dict) -> int:
        """Return the unique group_size across the config; raise if missing/mismatched."""
        sizes = set()

        def _add_group_size_from_dict(config: dict):
            group_size = config.get("group_size")
            if isinstance(group_size, int):
                sizes.add(group_size)

        # Top-level and 'quantization' block
        _add_group_size_from_dict(cfg)
        quantization = cfg.get("quantization")
        if isinstance(quantization, dict):
            _add_group_size_from_dict(quantization)

        # config_groups: accept group-level or nested dicts (e.g., weights/input_activations)
        for config_groups in (cfg.get("config_groups") or {}).values():
            if isinstance(config_groups, dict):
                _add_group_size_from_dict(config_groups)
                for config_group in config_groups.values():
                    if isinstance(config_group, dict):
                        _add_group_size_from_dict(config_group)

        if not sizes:
            raise ValueError("No group_size found in config.")
        if len(sizes) > 1:
            raise ValueError(f"Inconsistent group_size values: {sorted(sizes)}")
        return next(iter(sizes))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> ModelOptFp4Config:
        # Handle two different config formats:
        # 1. hf_quant_config.json format: {"quantization": {"quant_algo": "NVFP4", ...}}
        # 2. config.json quantization_config format: {"quant_algo": "NVFP4", ...}
        # In future modelopt will deprecate hf_quant_config.json, and only keep config.json.
        # For legacy reasons, we keep hf_quant_config.json for now.

        # Initialize variables
        group_size = None
        exclude_modules = []

        # Try flat format first (config.json quantization_config - preferred format)
        quant_method = config.get("quant_algo")
        if quant_method is not None:
            group_size = config.get("group_size")
            # If group_size is not at top level, try to extract from config_groups
            if group_size is None:
                config_groups = config.get("config_groups", {})
                if config_groups:
                    # Get group_size from the first group's weights config
                    first_group = next(iter(config_groups.values()), {})
                    weights_config = first_group.get("weights", {})
                    group_size = weights_config.get("group_size")

            exclude_modules = config.get("ignore", [])
        else:
            # Fall back to nested format (hf_quant_config.json - legacy format)
            try:
                quant_config = cls.get_from_keys(config, ["quantization"])
                quant_method = quant_config["quant_algo"]
                group_size = ModelOptFp4Config.common_group_size(config)
                exclude_modules = quant_config.get("exclude_modules", [])
            except (ValueError, KeyError):
                raise ValueError(
                    "Cannot find 'quant_algo' in the model's quantization config. "
                    "Expected either flat format (config.json) or nested format (hf_quant_config.json)."
                )

        if not quant_method in ["NVFP4"]:
            raise ValueError(
                f"ModelOpt currently only supports: NVFP4 quantization in sglang diffusion. The provided quant_algo is {quant_method}. Please check the "
                "quantization config for your model's configuration."
            )
        is_checkpoint_nvfp4_serialized = "NVFP4" in quant_method

        if group_size is None or exclude_modules is None:
            logger.warning(
                f"group_size: {group_size}," f"exclude_modules: {exclude_modules}"
            )
            raise ValueError(
                "NVFP4 quantization requires group_size and exclude_modules "
                "specified in the quantization config"
            )
        return cls(
            is_checkpoint_nvfp4_serialized,
            group_size,
            exclude_modules,
            config.get("packed_modules_mapping"),
        )

    def is_layer_excluded(self, prefix: str):
        import regex as re

        fused_patterns = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"]
        prefix_split = prefix.split(".")
        for pattern in self.exclude_modules:
            regex_str = pattern.replace(".", r"\.").replace("*", r".*")
            pattern_split = pattern.split(".")
            if re.fullmatch(regex_str, prefix):
                return True
            elif (
                pattern_split[-1] in fused_patterns
                and pattern_split[-1] in prefix_split[-1]
            ):
                # Check if the last part of the excluded pattern is contained in the last part of the prefix
                # This handles fused modules like fused_qkv_a_proj_with_mqa that contain q_a_proj and kv_a_proj_with_mqa
                # e.g., model.layers.{i}.self_attn.{fused_weight_name}
                assert len(prefix_split) == 5 and len(pattern_split) == 5
                return True
        return False

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        linear_cls = (
            ComfyUIFp4LinearMethod if _ck_available else ModelOptFp4LinearMethod
        )
        return self._get_quant_method(layer, prefix, Linear=linear_cls)


class ModelOptFp4LinearMethod(LinearMethodBase):
    """Linear method for NVFP4.
    Supports loading NVFP4 checkpoints with the following structure:

    |Tensor Name           | datatype      |  shape      |
    |----------------------------------------------------|
    |input_scale           | torch.float32 | scalar      |
    |weight                | NVFP4(SE2M1)  | [1, X, y/2] |
    |weight_scale          | FP8-E4M3      | [X, Y]      |
    |weight_scale_2        | torch.float32 | scalar      |

    The weights are quantized per block of 16 elements.
    Args: quant_config: The ModelOpt quantization config.
    """

    def __init__(self, quant_config: ModelOptFp4Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        if not self.quant_config.is_checkpoint_nvfp4_serialized:
            raise ValueError(
                "NVFP4 quantization was selected, "
                " dynamic quantization is not supported."
            )
        if input_size_per_partition % 16 != 0:
            raise ValueError(
                f"Unsupported model when input features size is {input_size_per_partition}, not multiple of 16, for NVFP4 quantization."
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_nvfp4_serialized
            else params_dtype
        )

        weight = ModelWeightParameter(
            data=torch.empty(
                # 2 fp4 data is packed in one uint8 in the input dimension
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        input_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )

        layer.register_parameter("input_scale", input_scale)

        weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale_2", weight_scale_2)

        weight_scale = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.group_size,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        input_scale_2 = layer.input_scale.max().to(torch.float32)
        weight_scale_2 = layer.weight_scale_2.max().to(torch.float32)

        copy_or_rebind_param(
            layer, "alpha", (input_scale_2 * weight_scale_2).to(torch.float32)
        )
        copy_or_rebind_param(
            layer, "input_scale_inv", (1 / input_scale_2).to(torch.float32)
        )

        # Store original output size before any padding
        layer.output_size_per_partition = layer.weight.shape[0]

        # Pad weights for CUTLASS/FlashInfer kernel alignment (K and N divisible by 32)
        weight, weights_padding_cols = pad_nvfp4_weight(layer.weight.data)
        layer.weights_padding_cols = weights_padding_cols
        copy_or_rebind_param(layer, "weight", weight)

        # Pad and blockwise interleave weight_scale
        scales = layer.weight_scale
        scale_ndim = scales.ndim
        if scale_ndim == 2:
            scales = scales.unsqueeze(0)
        assert scales.ndim == 3
        B, M, K = scales.shape
        M_padded = round_up_to_multiple(M, 128)
        K_padded = round_up_to_multiple(K, 4)
        padded_scales = torch.zeros((B, M_padded, K_padded), dtype=scales.dtype)
        padded_scales[:B, :M, :K] = scales
        batches, rows, cols = padded_scales.shape
        assert rows % 128 == 0
        assert cols % 4 == 0
        padded_scales = padded_scales.reshape(batches, rows // 128, 4, 32, cols // 4, 4)
        padded_scales = padded_scales.permute((0, 1, 4, 3, 2, 5))
        padded_scales = padded_scales.contiguous().cuda()
        padded_scales = (
            padded_scales.reshape(M_padded, K_padded)
            if scale_ndim == 2
            else padded_scales.reshape(B, M_padded, K_padded)
        )
        copy_or_rebind_param(layer, "weight_scale_interleaved", padded_scales)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output_dtype = x.dtype
        # Support arbitrary leading batch dimensions (e.g. [B, S, K] or [M, K])
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])

        # Get original output size (before padding) and padded weight size
        output_size = layer.output_size_per_partition
        w_n = layer.weight.shape[0]
        output_shape = list(input_shape[:-1]) + [output_size]

        # Quantize BF16 or FP16 to (FP4 and interleaved block scale)
        x_fp4, x_scale_interleaved = fp4_quantize(x, layer.input_scale_inv)

        assert x_fp4.dtype == torch.uint8
        assert layer.weight.dtype == torch.uint8
        assert layer.weight_scale_interleaved.dtype == torch.float8_e4m3fn
        assert layer.alpha.dtype == torch.float32

        # Pad activations to match weight K-dimension padding
        weights_padding_cols = getattr(layer, "weights_padding_cols", 0)
        x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)

        w = layer.weight
        w_scale_interleaved = layer.weight_scale_interleaved

        # Ensure block scales are in fp8 because the CUTLASS kernel rejects uint8
        if x_scale_interleaved.dtype == torch.uint8:
            x_scale_interleaved = x_scale_interleaved.view(torch.float8_e4m3fn)
        if w_scale_interleaved.dtype == torch.uint8:
            w_scale_interleaved = w_scale_interleaved.view(torch.float8_e4m3fn)
        if cutlass_fp4_gemm is None:
            raise RuntimeError(
                "sgl_kernel.cutlass_scaled_fp4_mm is not available. "
                "Install or upgrade sgl_kernel to use ModelOptFp4LinearMethod."
            )
        out = cutlass_fp4_gemm(
            x_fp4,
            w,
            x_scale_interleaved,
            w_scale_interleaved,
            layer.alpha,
            output_dtype,
        )

        # Slice output to remove N-dimension padding
        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


class ComfyUIFp4LinearMethod(LinearMethodBase):
    """Linear method for NVFP4 using comfy-kitchen (cuBLAS) kernels.

    Same checkpoint format as ModelOptFp4LinearMethod, but uses
    comfy_kitchen.scaled_mm_nvfp4 for the GEMM instead of flashinfer/CUTLASS.

    Alignment requirements (less strict than CUTLASS):
      - Weight N dimension: multiple of 8
      - Weight K dimension: multiple of 16

    Block scale format: cuBLAS swizzled layout
      [roundup(N, 128), roundup(K//16, 4)] in fp8_e4m3fn

    Requires SM >= 10.0 (Blackwell) for hardware-accelerated matmul.
    """

    def __init__(self, quant_config: ModelOptFp4Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        if not self.quant_config.is_checkpoint_nvfp4_serialized:
            raise ValueError(
                "NVFP4 quantization was selected, "
                "dynamic quantization is not supported."
            )
        if input_size_per_partition % 16 != 0:
            raise ValueError(
                f"Unsupported model when input features size is {input_size_per_partition}, "
                "not multiple of 16, for NVFP4 quantization."
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        input_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_scale", input_scale)

        weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale_2", weight_scale_2)

        weight_scale = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from comfy_kitchen.float_utils import from_blocked, to_blocked

        input_scale = layer.input_scale.max().to(torch.float32)
        weight_scale_2 = layer.weight_scale_2.max().to(torch.float32)

        # Store scalar per-tensor scales on CUDA for the kernel
        copy_or_rebind_param(layer, "input_scale_ck", input_scale.cuda())
        copy_or_rebind_param(layer, "weight_scale_2_ck", weight_scale_2.cuda())

        # Store original output size (N) before any padding
        layer.output_size_per_partition = layer.weight.shape[0]

        # Ensure weight is contiguous on CUDA
        copy_or_rebind_param(layer, "weight", layer.weight.data.contiguous().cuda())

        # The checkpoint's weight_scale is already in cuBLAS tiled layout —
        # modelopt quantizes using NVIDIA's cuBLAS FP4 kernel, which writes
        # block scales in the hardware-native tiled format directly.
        # Therefore we must NOT apply an additional swizzle (unlike the
        # flashinfer path which converts cuBLAS-tiled → CUTLASS format).
        #
        # cuBLAS tiled format shape requirement:
        #   (roundup(N, 128), roundup(K//16, 4))
        #
        # If dimensions are already aligned (common for FLUX), use as-is.
        # Otherwise: unswizzle → pad → re-swizzle so the padding zeros end
        # up in the correct tiled positions.
        scales = layer.weight_scale.data  # [N, K//16] fp8_e4m3fn
        N, Ks = scales.shape
        N_padded = round_up_to_multiple(N, 128)
        Ks_padded = round_up_to_multiple(Ks, 4)

        if N == N_padded and Ks == Ks_padded:
            # Already aligned — use the checkpoint block scales directly
            weight_scale_ck = scales.cuda()
        else:
            # Unswizzle (cuBLAS tiled → row-major), zero-pad, re-swizzle
            scales_rm = from_blocked(scales, num_rows=N, num_cols=Ks)
            padded_rm = torch.zeros((N_padded, Ks_padded), dtype=scales.dtype)
            padded_rm[:N, :Ks] = scales_rm
            weight_scale_ck = to_blocked(padded_rm, flatten=False).cuda()

        copy_or_rebind_param(layer, "weight_scale_ck", weight_scale_ck)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not _ck_available:
            raise RuntimeError(
                "comfy_kitchen is not available. "
                "Install it to use ModelOptFp4CKLinearMethod."
            )

        output_dtype = x.dtype
        input_shape = x.shape
        x_2d = x.view(-1, input_shape[-1])  # [M, K]
        M = x_2d.shape[0]

        output_size = layer.output_size_per_partition
        output_shape = list(input_shape[:-1]) + [output_size]

        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()

        # Quantize BF16/FP16 activation to NVFP4 with cuBLAS block-scale layout.
        # pad_16x=True pads M and K to multiples of 16 (K already satisfies this).
        # Returns:
        #   x_fp4:        [roundup(M, 16), K//2]                  uint8
        #   x_block_scale:[roundup(roundup(M,16), 128), roundup(K//16, 4)]  fp8_e4m3fn
        x_fp4, x_block_scale = ck_cuda.quantize_nvfp4(
            x_2d, layer.input_scale_ck, pad_16x=True
        )

        # cuBLAS NVFP4 GEMM: computes (x_fp4 @ weight.T) * alpha + bias
        out = ck_cuda.scaled_mm_nvfp4(
            x_fp4,
            layer.weight,
            tensor_scale_a=layer.input_scale_ck,
            tensor_scale_b=layer.weight_scale_2_ck,
            block_scale_a=x_block_scale,
            block_scale_b=layer.weight_scale_ck,
            bias=bias,
            out_dtype=output_dtype,
        )
        # out: [roundup(M, 16), N] — slice back to original [M, output_size]
        out = out[:M, :output_size].contiguous()

        return out.view(*output_shape)
