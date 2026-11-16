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
    m = vit_b_16(weights=None).eval()
    dummy = torch.randn(batch_size, 3, input_resolution, input_resolution)

    tmp_path = output_path + ".tmp"
    torch.onnx.export(
        m, dummy, tmp_path,
        opset_version=opset_version,
        input_names=["input"], output_names=["output"],
        do_constant_folding=True,
    )

    model = shape_inference.infer_shapes(onnx.load(tmp_path))
    onnx.save(model, output_path)
    os.remove(tmp_path)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[ViT-B/16] exported to {output_path} ({size_mb:.1f} MB)")

    from collections import Counter
    c = Counter(n.op_type for n in model.graph.node)
    print(f"[ViT-B/16] op distribution: {dict(c)}")

    return output_path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    os.chdir(repo_root)
    export_vit_b_16()
