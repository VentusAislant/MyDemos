import torch
from torch import nn
import functools
from torch.distributed.fsdp import MixedPrecision

# requires grad scaler in main loop
fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float16,
    # Buffer precision.
    buffer_dtype=torch.float16,
)

bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)

bfSixteen_working = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

fp32_policy = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)


def get_wrapper():
    def _100M_auto_wrap_policy(
            module: nn.Module,
            recurse: bool,
            nonwrapped_numel: int,
            ## additional arguments
            min_num_params: int = int(1e8),  # 100M
    ) -> bool:
        return nonwrapped_numel >= min_num_params

    auto_wrap_policy = functools.partial(
        _100M_auto_wrap_policy,
        min_num_params=int(1e8)
    )

    return auto_wrap_policy
