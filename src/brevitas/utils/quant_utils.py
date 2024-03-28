# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.core.bit_width import BitWidthParameter
from brevitas.core.function_wrapper import *
from brevitas.core.quant import RescalingIntQuant
from brevitas.function.ops import max_int
from brevitas.function.ops import min_int
from brevitas.inject.enum import FloatToIntImplType


def has_learned_weight_bit_width(module):
    from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector

    if isinstance(module, WeightQuantProxyFromInjector) \
            and isinstance(module.tensor_quant, RescalingIntQuant) \
            and isinstance(module.tensor_quant.msb_clamp_bit_width_impl,
                           BitWidthParameter):
        return True
    else:
        return False


def has_learned_activation_bit_width(module):
    from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
    from brevitas.proxy.runtime_quant import FusedActivationQuantProxy

    if isinstance(module, ActQuantProxyFromInjector) \
            and isinstance(module.fused_activation_quant_proxy, FusedActivationQuantProxy) \
            and isinstance(module.fused_activation_quant_proxy.tensor_quant, RescalingIntQuant) \
            and isinstance(module.fused_activation_quant_proxy.tensor_quant.msb_clamp_bit_width_impl,
                           BitWidthParameter):
        return True
    else:
        return False


def float_to_int_impl_to_enum(module):
    if isinstance(module, RoundSte):
        return FloatToIntImplType.ROUND
    elif isinstance(module, RoundToZeroSte):
        return FloatToIntImplType.ROUND_TO_ZERO
    elif isinstance(module, FloorSte):
        return FloatToIntImplType.FLOOR
    elif isinstance(module, CeilSte):
        return FloatToIntImplType.CEIL
    elif isinstance(module, DPURoundSte):
        return FloatToIntImplType.DPU
    elif isinstance(module, LearnedRoundSte):
        return FloatToIntImplType.LEARNED_ROUND
    elif isinstance(module, StochasticRoundSte):
        if module.deterministic_inference:
            return FloatToIntImplType.ROUND
        else:
            return FloatToIntImplType.STOCHASTIC_ROUND
    else:
        return None


def int_clip_symbolic_kwargs(narrow, signed, bit_width) -> dict:
    # equality comparisons among power-of-2 floats are okay
    if narrow or bit_width != 8. and bit_width != 32.:
        if signed and (bit_width < 8. or narrow and bit_width <= 8.):
            dtype = torch.int8
        elif not signed and (bit_width < 8. or narrow and bit_width <= 8.):
            dtype = torch.uint8
        elif signed and (bit_width < 32. or narrow and bit_width <= 32.):
            dtype = torch.int32
        else:
            raise RuntimeError(f"Sign {signed} and bit width {bit_width} not supported for export.")
        return {
            'min_val': min_int(signed, narrow, bit_width).to(dtype).item(),
            'max_val': max_int(signed, narrow, bit_width).to(dtype).item()}
    else:
        return None


def quant_axis(scale):
    for i, s in enumerate(scale.shape):
        if s != 1:
            return i
    return None


def zero_point_with_dtype(signed, bit_width, zero_point):
    if not signed:
        if (zero_point < 0).any():
            raise RuntimeError("Zero points have to be positive under unsigned quantization")
        if bit_width > 8:
            raise RuntimeError("Unsigned zero-point with bit-width > 8 not supported.")
        return zero_point.type(torch.uint8)
    else:
        if bit_width <= 8:
            return zero_point.type(torch.int8)
        else:
            return zero_point.type(torch.int32)
