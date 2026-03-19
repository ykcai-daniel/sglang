# SPDX-License-Identifier: Apache-2.0

import glob
import os

from sglang.multimodal_gen.runtime.pipelines.flux_2 import Flux2Pipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_FLUX2_BASE_MODEL = "black-forest-labs/FLUX.2-dev"


def _find_mixed_safetensors(local_dir: str) -> str | None:
    """Return the path to the *-mixed.safetensors file in a directory, or None."""
    mixed_files = sorted(glob.glob(os.path.join(local_dir, "*-mixed.safetensors")))
    return mixed_files[0] if mixed_files else None


class Flux2NvfpPipeline(Flux2Pipeline):
    pipeline_name = "Flux2NvfpPipeline"

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict | None = None,
    ) -> dict:
        nvfp4_path = self.model_path
        server_args.model_path = _FLUX2_BASE_MODEL
        self.model_path = _FLUX2_BASE_MODEL
        logger.info(
            "Using base model '%s' for non-transformer components", _FLUX2_BASE_MODEL
        )

        if server_args.transformer_weights_path is None:
            local_nvfp4_path = maybe_download_model(nvfp4_path)
            mixed_file = _find_mixed_safetensors(local_nvfp4_path)
            if mixed_file:
                logger.info("Using mixed-precision NVFP4 weights: %s", mixed_file)
                server_args.transformer_weights_path = mixed_file
            else:
                logger.warning(
                    "No *-mixed.safetensors found in %s; falling back to full directory",
                    local_nvfp4_path,
                )
                server_args.transformer_weights_path = local_nvfp4_path

        logger.info(
            "NVFP4 transformer weights: %s", server_args.transformer_weights_path
        )
        return super().load_modules(server_args, loaded_modules)


EntryClass = Flux2NvfpPipeline
