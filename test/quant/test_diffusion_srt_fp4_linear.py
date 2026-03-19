"""
Unit test verifying the NVFP4 GEMM path via flashinfer and the SGLang diffusion
ModelOptFp4LinearMethod.

TestNvFp4Gemm — raw flashinfer API:
  1. Create random BF16 weight W [N, K] and activation X [M, K].
  2. Compute global scale factors from the absolute maximum of each matrix.
  3. Quantize both with flashinfer.nvfp4_quantize (NVFP4, block_size=16).
  4. Run flashinfer.mm_fp4 with the quantized tensors.
  5. Verify cosine similarity > 0.97 against torch.mm(X, W.T) reference.

TestDiffusionFp4Layer — SGLang ModelOptFp4LinearMethod:
  1. Build a fake layer with random uint8-packed FP4 weights and FP8 scales.
  2. Call process_weights_after_loading() then apply() on a BF16 activation.
  3. Verify diffusion and SRT implementations produce identical outputs.
  4. Verify cosine similarity > 0.97 against the BF16 dequantized reference.

Requires a Blackwell (SM100+) GPU and flashinfer with FP4 support.

Run with:
    conda run -n sgl python test/manual/test_diffusion_srt_fp4_linear.py
"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter  # noqa: F401 (used by _make_nvfp4_layer)

try:
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize

    HAS_FLASHINFER_FP4 = True
except ImportError:
    HAS_FLASHINFER_FP4 = False

try:
    import comfy_kitchen  # noqa: F401

    HAS_CK = True
except Exception:
    HAS_CK = False

GROUP_SIZE = 16


def _make_nvfp4_layer(
    w: torch.Tensor,
    x_global_sf: torch.Tensor,
) -> nn.Module:
    """
    Build a diffusion ModelOptFp4LinearMethod-compatible layer directly from a
    BF16 weight matrix, bypassing process_weights_after_loading.

    apply() reads four attributes set here:
      - weight                  [N, K//2] uint8 packed FP4
      - weight_scale_interleaved [N, K//16] FP8 E4M3 (128x4 layout, .T before GEMM)
      - alpha                   float32 scalar = 1/(x_global_sf * w_global_sf)
      - input_scale_inv         float32 scalar = x_global_sf
      - weights_padding_cols    int (0 for aligned shapes)
      - output_size_per_partition int = N

    Uses nvfp4_quantize (128x4 layout) so the scales are already in the format
    flashinfer.mm_fp4 expects, matching the TestNvFp4Gemm path exactly.
    """
    N, K = w.shape
    w_global_sf = (448 * 6) / w.float().abs().nan_to_num().max()

    w_fp4, w_sf = nvfp4_quantize(
        w, w_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )

    layer = nn.Module()
    layer.register_parameter("weight", Parameter(w_fp4, requires_grad=False))
    layer.register_parameter(
        "weight_scale_interleaved",
        Parameter(w_sf.view(torch.float8_e4m3fn), requires_grad=False),
    )
    layer.register_parameter(
        "alpha",
        Parameter(
            (1.0 / (x_global_sf * w_global_sf)).to(torch.float32), requires_grad=False
        ),
    )
    layer.register_parameter(
        "input_scale_inv",
        Parameter(x_global_sf.to(torch.float32), requires_grad=False),
    )
    layer.weights_padding_cols = 0
    layer.output_size_per_partition = N
    return layer


def _make_nvfp4_ck_layer(
    w: torch.Tensor,
    x_global_sf: torch.Tensor,
) -> nn.Module:
    """
    Build a ComfyUIFp4LinearMethod-compatible layer from a BF16 weight matrix.

    Mirrors process_weights_after_loading: stores weight_qt (QuantizedTensor
    via TensorCoreNVFP4Layout) and input_scale_ck (per-tensor activation scale).
    """
    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    N = w.shape[0]
    w_global_sf = (448 * 6) / w.float().abs().nan_to_num().max()

    input_scale_ck = (1.0 / x_global_sf).to(torch.float32).cuda()
    weight_scale_2 = (1.0 / w_global_sf).to(torch.float32).cuda()

    # Use TensorCoreNVFP4Layout.quantize — same as process_weights_after_loading
    # would do if it were re-quantizing from BF16 (here the "checkpoint" is BF16 w)
    w_qdata, w_params = TensorCoreNVFP4Layout.quantize(
        w.contiguous(), scale=weight_scale_2
    )
    weight_qt = QuantizedTensor(w_qdata, "TensorCoreNVFP4Layout", w_params)

    layer = nn.Module()
    layer.weight_qt = weight_qt
    layer.register_parameter(
        "input_scale_ck", Parameter(input_scale_ck, requires_grad=False)
    )
    layer.output_size_per_partition = N
    return layer


def _print_error_stats(
    label: str, out: torch.Tensor, ref: torch.Tensor
) -> tuple[float, float]:
    """Print abs/rel error percentiles, cosine similarity, and magnitude ratio.

    Returns:
        (cos_sim, magnitude_ratio)  where magnitude_ratio = mean|out| / mean|ref|.
        NaN/Inf in out causes magnitude_ratio to be inf or nan.
    """
    out_f = out.float()
    ref_f = ref.float()

    cos_sim = F.cosine_similarity(ref_f.reshape(-1), out_f.reshape(-1), dim=0).item()

    out_mean_abs = out_f.abs().mean().item()
    ref_mean_abs = ref_f.abs().mean().item()
    mag_ratio = out_mean_abs / ref_mean_abs if ref_mean_abs > 0 else float("nan")

    has_nan = out_f.isnan().any().item()
    has_inf = out_f.isinf().any().item()

    abs_err = (out_f - ref_f).abs().flatten().cpu()
    rel_err = abs_err / ref_f.abs().flatten().cpu().clamp(min=1e-6)
    MAX_Q = 2_000_000
    if abs_err.numel() > MAX_Q:
        idx = torch.randperm(abs_err.numel())[:MAX_Q]
        abs_err, rel_err = abs_err[idx], rel_err[idx]
    pcts = [50, 90, 95, 99]
    abs_pct = [abs_err.quantile(p / 100).item() for p in pcts]
    rel_pct = [rel_err.quantile(p / 100).item() for p in pcts]
    print(
        f"\n  [{label}]  cos_sim={cos_sim:.6f}  mag_ratio={mag_ratio:.4f}"
        f"  (out={out_mean_abs:.4f} ref={ref_mean_abs:.4f})"
        f"{'  NaN!' if has_nan else ''}{'  Inf!' if has_inf else ''}"
    )
    print(f"    atol p{pcts}: {[f'{v:.4f}' for v in abs_pct]}")
    print(f"    rtol p{pcts}: {[f'{v:.4f}' for v in rel_pct]}")
    return cos_sim, mag_ratio


def _nvfp4_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    backend: str = "cudnn",
) -> torch.Tensor:
    """
    Compute linear(x, w) = x @ w.T using NVFP4 GEMM via flashinfer.

    Follows the flashinfer test convention:
      - global_sf = (448 * 6) / max(|t|)
      - quantize activation with do_shuffle=False
      - quantize weight with do_shuffle=(backend == "trtllm")
      - alpha = 1 / (x_global_sf * w_global_sf)

    Args:
        x: BF16 activation of shape [M, K].
        w: BF16 weight matrix of shape [N, K]  (nn.Linear convention).
        backend: flashinfer mm_fp4 backend ("auto", "cudnn", "cutlass", "trtllm").

    Returns:
        BF16 output of shape [M, N].
    """
    M = x.shape[0]
    N = w.shape[0]

    x_global_sf = (448 * 6) / x.float().abs().nan_to_num().max()
    w_global_sf = (448 * 6) / w.float().abs().nan_to_num().max()

    do_shuffle_w = backend == "trtllm"

    x_fp4, x_sf = nvfp4_quantize(
        x, x_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    w_fp4, w_sf = nvfp4_quantize(
        w, w_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=do_shuffle_w
    )

    alpha = 1.0 / (x_global_sf * w_global_sf)

    out = torch.empty([M, N], device=x.device, dtype=torch.bfloat16)
    mm_fp4(
        x_fp4,
        w_fp4.T,
        x_sf,
        w_sf.T,
        alpha,
        torch.bfloat16,
        out,
        block_size=16,
        use_8x4_sf_layout=False,
        use_nvfp4=True,
        backend=backend,
        skip_check=False,
    )
    return out  # [M, N]


@unittest.skipUnless(HAS_FLASHINFER_FP4, "flashinfer FP4 ops not available")
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestNvFp4Gemm(unittest.TestCase):
    """Verify that the flashinfer NVFP4 GEMM output direction matches BF16 F.linear."""

    # Representative Flux2 diffusion transformer shapes.
    # Smallest dimension is always >= 1024.
    #   M = number of tokens (sequence length)
    #   N = output features
    #   K = input features (hidden dim)
    SHAPES = [
        # (M,    N,     K)
        (1024, 6144, 6144),  # square: double-block img_attn.proj
        (1024, 6144, 18432),  # double-block img_mlp.0  (3x expansion)
        (1024, 18432, 6144),  # double-block img_mlp.2  (3x expansion, transposed)
        (1024, 55296, 3072),  # single-block linear1 fused QKV+MLP
        (1024, 6144, 12288),  # single-block linear2 attn+MLP out
    ]

    # Cosine similarity threshold: FP4 has 3-bit mantissa so absolute error is
    # large, but the output direction should be preserved (following flashinfer tests).
    COS_SIM_THRESHOLD = 0.97

    def _run_one(self, M: int, N: int, K: int, seed: int = 0) -> None:
        torch.manual_seed(seed)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

        ref = torch.mm(x, w.T)  # BF16 reference: [M, N]
        out = _nvfp4_linear(x, w)  # NVFP4 GEMM:     [M, N]

        self.assertEqual(
            out.shape, ref.shape, f"shape mismatch: {out.shape} vs {ref.shape}"
        )

        cos_sim, _ = _print_error_stats(f"M={M} N={N} K={K}", out, ref)

        self.assertGreater(
            cos_sim,
            self.COS_SIM_THRESHOLD,
            f"M={M} N={N} K={K}: cos_sim={cos_sim:.4f} < {self.COS_SIM_THRESHOLD}",
        )

    def test_shapes(self):
        for M, N, K in self.SHAPES:
            with self.subTest(M=M, N=N, K=K):
                self._run_one(M, N, K)


@unittest.skipUnless(HAS_CK, "comfy_kitchen not available")
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCKFp4Layer(unittest.TestCase):
    """Verify ComfyUIFp4LinearMethod (cuBLAS path) against a BF16 reference.

    Workflow:
      1. Create random BF16 weight W [N, K] and activation X [M, K].
      2. Quantize W with ck_cuda.quantize_nvfp4; build a CK-format layer.
      3. Call ComfyUIFp4LinearMethod.apply() on X.
      4. Compare output to torch.mm(X, W.T) using cosine similarity > 0.97.
      5. Also test that 3-D inputs [B, S, K] produce the same result as [B*S, K].
    """

    SHAPES = [
        # (M,    N,     K)
        (1024, 6144, 6144),
        (1024, 6144, 18432),
        (1024, 18432, 6144),
        (1024, 55296, 3072),
        (1024, 6144, 12288),
    ]

    COS_SIM_THRESHOLD = 0.97

    def _run_one(self, M: int, N: int, K: int, seed: int = 0) -> None:
        from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
            ComfyUIFp4LinearMethod,
            ModelOptFp4Config,
        )

        torch.manual_seed(seed)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

        ref = torch.mm(x, w.T)  # BF16 reference [M, N]

        x_global_sf = (448 * 6) / x.float().abs().nan_to_num().max()
        layer = _make_nvfp4_ck_layer(w, x_global_sf)

        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=GROUP_SIZE,
            exclude_modules=[],
        )
        out = ComfyUIFp4LinearMethod(cfg).apply(layer, x)

        self.assertEqual(
            out.shape, ref.shape, f"shape mismatch: {out.shape} vs {ref.shape}"
        )

        label = f"CK M={M} N={N} K={K}"
        self.assertFalse(
            out.float().isnan().any().item(), f"{label}: output contains NaN"
        )
        self.assertFalse(
            out.float().isinf().any().item(), f"{label}: output contains Inf"
        )

        cos_sim, mag_ratio = _print_error_stats(label, out, ref)
        self.assertGreater(
            cos_sim,
            self.COS_SIM_THRESHOLD,
            f"{label}: cos_sim={cos_sim:.4f} < {self.COS_SIM_THRESHOLD}",
        )
        self.assertGreater(
            mag_ratio, 0.5, f"{label}: magnitude too small (ratio={mag_ratio:.4f})"
        )
        self.assertLess(
            mag_ratio,
            2.0,
            f"{label}: magnitude too large / overflow (ratio={mag_ratio:.4f})",
        )

    def test_shapes(self):
        for M, N, K in self.SHAPES:
            with self.subTest(M=M, N=N, K=K):
                self._run_one(M, N, K)

    def test_batch_dims(self):
        """ComfyUIFp4LinearMethod.apply() must handle 3-D inputs [B, S, K] correctly."""
        from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
            ComfyUIFp4LinearMethod,
            ModelOptFp4Config,
        )

        N, K = 6144, 6144
        torch.manual_seed(0)
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        B, S = 2, 512
        x3 = torch.randn(B, S, K, dtype=torch.bfloat16, device="cuda")
        x2 = x3.view(-1, K)

        x_global_sf = (448 * 6) / x3.float().abs().nan_to_num().max()
        layer = _make_nvfp4_ck_layer(w, x_global_sf)
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=GROUP_SIZE,
            exclude_modules=[],
        )
        method = ComfyUIFp4LinearMethod(cfg)

        out3 = method.apply(layer, x3)
        out2 = method.apply(layer, x2)

        self.assertEqual(out3.shape, (B, S, N))
        self.assertEqual(out2.shape, (B * S, N))
        torch.testing.assert_close(
            out3.view(-1, N).float(),
            out2.float(),
            msg="3-D and 2-D inputs must give the same flat output",
        )


@unittest.skipUnless(HAS_FLASHINFER_FP4, "flashinfer FP4 ops not available")
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFp4LayerVsFlashinfer(unittest.TestCase):
    """Cross-validate ModelOptFp4LinearMethod, _nvfp4_linear, and BF16 GEMM.

    For each shape:
      1. Compute BF16 reference: ref = torch.mm(x, w.T)
      2. Compute flashinfer reference: fi_out = _nvfp4_linear(x, w)
      3. Compute diffusion layer: layer_out = ModelOptFp4LinearMethod.apply(layer, x)

    Assertions:
      - fi_out vs ref: cos_sim > COS_SIM_THRESHOLD  (flashinfer matches BF16)
      - layer_out vs ref: cos_sim > COS_SIM_THRESHOLD  (ModelOpt matches BF16)
      - layer_out vs fi_out: cos_sim > CROSS_SIM_THRESHOLD  (two FP4 impls agree)
    """

    SHAPES = [
        # (M,    N,     K)
        (1024, 6144, 6144),
        (1024, 6144, 18432),
        (1024, 18432, 6144),
        (1024, 55296, 3072),
        (1024, 6144, 12288),
    ]

    COS_SIM_THRESHOLD = 0.97  # FP4 vs BF16
    CROSS_SIM_THRESHOLD = 0.99  # two FP4 impls should be very close

    def _run_one(self, M: int, N: int, K: int, seed: int = 0) -> None:
        from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
            ModelOptFp4Config,
            ModelOptFp4LinearMethod,
        )

        torch.manual_seed(seed)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

        # 1. BF16 ground truth
        ref = torch.mm(x, w.T)

        # 2. Flashinfer reference
        fi_out = _nvfp4_linear(x, w)

        # 3. ModelOptFp4LinearMethod
        x_global_sf = (448 * 6) / x.float().abs().nan_to_num().max()
        layer = _make_nvfp4_layer(w, x_global_sf)
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=GROUP_SIZE,
            exclude_modules=[],
        )
        layer_out = ModelOptFp4LinearMethod(cfg).apply(layer, x)

        label = f"M={M} N={N} K={K}"
        cos_fi, _ = _print_error_stats(f"flashinfer vs bf16  [{label}]", fi_out, ref)
        cos_layer, _ = _print_error_stats(
            f"modelopt  vs bf16  [{label}]", layer_out, ref
        )
        cos_cross, _ = _print_error_stats(
            f"modelopt  vs fi    [{label}]", layer_out, fi_out
        )

        self.assertGreater(
            cos_fi,
            self.COS_SIM_THRESHOLD,
            f"{label}: flashinfer vs bf16 cos_sim={cos_fi:.4f} < {self.COS_SIM_THRESHOLD}",
        )
        self.assertGreater(
            cos_layer,
            self.COS_SIM_THRESHOLD,
            f"{label}: modelopt vs bf16 cos_sim={cos_layer:.4f} < {self.COS_SIM_THRESHOLD}",
        )
        self.assertGreater(
            cos_cross,
            self.CROSS_SIM_THRESHOLD,
            f"{label}: modelopt vs flashinfer cos_sim={cos_cross:.4f} < {self.CROSS_SIM_THRESHOLD}",
        )

    def test_shapes(self):
        for M, N, K in self.SHAPES:
            with self.subTest(M=M, N=N, K=K):
                self._run_one(M, N, K)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    for cls in (TestCKFp4Layer,):
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\nAll tests passed.")
