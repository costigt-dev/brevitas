# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.export import export_onnx_qcdq
import brevitas.nn as qnn
from brevitas.quant.experimental.scaled_float import Fp8e4m3OCPWeightPerTensorFloat


def test_simple_fp8_export():
    model = qnn.QuantLinear(3, 16, weight_quant=Fp8e4m3OCPWeightPerTensorFloat)
    export_onnx_qcdq(model, torch.randn(1, 3), 'test.onnx', export_weight_q_node=True)
    assert True


if __name__ == "__main__":
    test_simple_fp8_export()
    print("Done")