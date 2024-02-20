# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC

import torch

from brevitas.export.common.handler.base import BaseHandler
from brevitas.export.common.handler.qcdq import CDQCastBiasQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import CDQCastMixin
from brevitas.export.common.handler.qcdq import DQCastMixin
from brevitas.export.common.handler.qcdq import QCDQCastActQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQCastDecoupledWeightQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import \
    QCDQCastDecoupledWeightQuantWithInputProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQCastTruncQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQCastWeightQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QMixin


def _itemize_clip_bounds(clip_args):
    if clip_args is not None:
        clip_args['min_val'] = clip_args['min_val'].item()
        clip_args['max_val'] = clip_args['max_val'].item()
    return clip_args


class TorchDQCastMixin(DQCastMixin, ABC):

    def __init__(self) -> None:
        super().__init__()
        self.symbolic_kwargs = {}

    def dequantize_fn(self, x, scale, zero_point, axis):
        # cast zero_point to float, otherwise if both x
        # and zero_point are uint (as in asym quant)
        # uint - uint can lead to errors. Don't cast x to float
        # as the main float datatype might not be float32 (e.g float16)
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.to(torch.float)
        else:
            zero_point = float(zero_point)
        return (x - zero_point) * scale

    def cast_fn(self, x, dtype):
        return x.type(dtype)

    @property
    def flatten_dequantize_params(self):
        return False

    @property
    def itemize_quantize_scalar_params(self):
        return True

    def validate(self, module):
        assert module.bit_width() > 1., 'Binary quant not supported'


class TorchCDQCastMixin(CDQCastMixin, TorchDQCastMixin, ABC):

    def clip_fn(self, x, min_val, max_val):
        return torch.clamp(x, min_val, max_val)


class TorchQCDQCastMixin(QMixin, TorchCDQCastMixin, ABC):

    @classmethod
    def int8_dtype(cls):
        return torch.qint8

    @classmethod
    def uint8_dtype(cls):
        return torch.quint8

    @classmethod
    def int32_dtype(cls):
        return torch.qint32

    def validate(self, module):
        super().validate(module)
        if getattr(self, '_export_q_node', True):
            assert module.rounding_mode.upper() == 'ROUND', 'Only round to nearest even supported'
        assert not module.is_groupwise, "Export with Per Group quantization not supported"

    def quantize_fn(self, x, scale, zero_point, dtype, axis):
        if axis is None:
            y = torch.quantize_per_tensor(x, scale, zero_point, dtype)
        else:
            y = torch.quantize_per_channel(x, scale, zero_point, axis, dtype)
        return y.int_repr()


class TorchQCDQHandler(BaseHandler):

    def forward(self, *args, **kwargs):
        return self.symbolic_execution(*args, **kwargs)


class TorchQCDQCastWeightQuantProxyHandler(TorchQCDQCastMixin,
                                           QCDQCastWeightQuantProxyHandlerMixin,
                                           TorchQCDQHandler):
    _export_q_node = False

    @classmethod
    def int_clip_symbolic_kwargs(cls, narrow, signed, bit_width):
        clip_args = super().int_clip_symbolic_kwargs(narrow, signed, bit_width)
        return _itemize_clip_bounds(clip_args)


class TorchQCDQCastDecoupledWeightQuantProxyHandler(TorchQCDQCastMixin,
                                                    QCDQCastDecoupledWeightQuantProxyHandlerMixin,
                                                    TorchQCDQHandler):
    _export_q_node = False

    @classmethod
    def int_clip_symbolic_kwargs(cls, narrow, signed, bit_width):
        clip_args = super().int_clip_symbolic_kwargs(narrow, signed, bit_width)
        return _itemize_clip_bounds(clip_args)


class TorchQCDQCastDecoupledWeightQuantWithInputProxyHandler(
        TorchQCDQCastMixin, QCDQCastDecoupledWeightQuantWithInputProxyHandlerMixin,
        TorchQCDQHandler):
    _export_q_node = False

    @classmethod
    def int_clip_symbolic_kwargs(cls, narrow, signed, bit_width):
        clip_args = super().int_clip_symbolic_kwargs(narrow, signed, bit_width)
        return _itemize_clip_bounds(clip_args)


class TorchQCDQCastActQuantProxyHandler(TorchQCDQCastMixin,
                                        QCDQCastActQuantProxyHandlerMixin,
                                        TorchQCDQHandler):

    @classmethod
    def int_clip_symbolic_kwargs(cls, narrow, signed, bit_width):
        clip_args = super().int_clip_symbolic_kwargs(narrow, signed, bit_width)
        return _itemize_clip_bounds(clip_args)


class TorchCDQCastBiasQuantProxyHandler(TorchDQCastMixin,
                                        CDQCastBiasQuantProxyHandlerMixin,
                                        TorchQCDQHandler):
    pass


class TorchQCDQCastTruncQuantProxyHandler(TorchQCDQCastMixin,
                                          QCDQCastTruncQuantProxyHandlerMixin,
                                          TorchQCDQHandler):

    @classmethod
    def int_clip_symbolic_kwargs(cls, narrow, signed, bit_width):
        clip_args = super().int_clip_symbolic_kwargs(narrow, signed, bit_width)
        return _itemize_clip_bounds(clip_args)
