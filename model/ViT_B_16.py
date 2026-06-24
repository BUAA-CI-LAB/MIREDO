"""ViT-Base/16 ONNX 导出脚本。

运行该脚本会生成 model/vit_b_16.onnx（shape-inferred）。

注意：该脚本依赖 torch + torchvision（不在 MIREDO conda env，需要 pim env 或其他带 torch 的环境）。
MIREDO 运行时只读 ONNX 文件，不需要 torch。

用法：
    /home/xiaolin/anaconda3/envs/pim/bin/python model/ViT_B_16.py
"""

import os
import sys
import onnx
import torch
from onnx import shape_inference
from torchvision.models import vit_b_16


def export_vit_b_16(output_path: str = "model/vit_b_16.onnx",
                    input_resolution: int = 224,
                    batch_size: int = 1,
                    opset_version: int = 13) -> str:
    """Export ViT-Base/16 (86M params) to ONNX with shape inference.

    - Weights are random (weights=None); we only need shape info for mapping.
    - batch=1, 224×224 → seq_len = 197 (196 patches + 1 CLS token), d_model = 768.
    """
    m = vit_b_16(weights=None).eval()
    dummy = torch.randn(batch_size, 3, input_resolution, input_resolution)

    tmp_path = output_path + ".tmp"
    torch.onnx.export(
        m, dummy, tmp_path,
        opset_version=opset_version,
        input_names=["input"], output_names=["output"],
        do_constant_folding=True,
    )

    # Shape inference makes per-tensor dims explicit for our parser
    model = shape_inference.infer_shapes(onnx.load(tmp_path))
    onnx.save(model, output_path)
    os.remove(tmp_path)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[ViT-B/16] exported to {output_path} ({size_mb:.1f} MB)")

    # Quick node-type report
    from collections import Counter
    c = Counter(n.op_type for n in model.graph.node)
    print(f"[ViT-B/16] op distribution: {dict(c)}")

    return output_path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    os.chdir(repo_root)
    export_vit_b_16()
