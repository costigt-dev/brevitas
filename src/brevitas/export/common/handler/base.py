# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
import math

import torch
from torch import Tensor
from torch.nn import Module

__all__ = ['BaseHandler']


class BaseHandler(Module, ABC):

    def __init__(self) -> None:
        super().__init__()

    def attach_debug_info(self, module):
        pass

    @abstractmethod
    def prepare_for_export(self, module):
        pass


class ScaleHandlerMixin(ABC):

    @classmethod
    def validate_scalar_scale(cls, scale: Tensor):
        if scale is None:
            raise RuntimeError("Scale cannot be None.")
        if scale.view(-1).shape[0] != 1:
            raise RuntimeError("Only per-tensor scaling is supported.")
        return scale.item()

    @classmethod
    def validate_scalar_int_exponent(cls, scale: Tensor):
        cls.validate_scalar_scale(scale)
        exponent = math.log2(scale)
        if not exponent.is_integer():
            raise RuntimeError("Only power-of-two scale factors are supported.")
        exponent = int(exponent)
        return exponent

    @classmethod
    def validate_neg_scalar_int_exponent(cls, scale: Tensor):
        return -cls.validate_scalar_int_exponent(scale)
