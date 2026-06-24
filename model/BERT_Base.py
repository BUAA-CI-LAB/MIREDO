"""BERT-Base-uncased ONNX 导出脚本。

生成 model/bert_base.onnx（shape-inferred + onnx-simplifier 处理）。
MIREDO 运行时仅读 ONNX 文件。

BERT-Base-uncased: 12 encoder layers, d_model=768, d_head=64, heads=12,
MLP hidden=3072. 我们用 seq_len=128, batch=1 作为典型评估输入。

用法：
    /home/xiaolin/anaconda3/envs/pim/bin/python model/BERT_Base.py
"""

import os
import onnx
import torch
from onnx import shape_inference


def export_bert_base(output_path: str = "model/bert_base.onnx",
                     seq_len: int = 128,
                     batch_size: int = 1,
                     opset_version: int = 13,
                     simplify: bool = True) -> str:
    """Export BERT-Base-uncased to ONNX, seq=128, batch=1, random weights."""
    # 随机初始化权重（形状是全部信息），避免下载 ~440 MB pretrained 权重
    from transformers import BertConfig, BertModel
    cfg = BertConfig()  # bert-base 的默认配置就是我们要的
    m = BertModel(cfg).eval()

    # 输入：input_ids (long) + attention_mask (long)
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    tmp_path = output_path + ".tmp"
    # 显式列出 inputs/outputs，避免命名冲突
    torch.onnx.export(
        m,
        (input_ids, attention_mask),
        tmp_path,
        opset_version=opset_version,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state", "pooler_output"],
        do_constant_folding=True,
    )

    model = shape_inference.infer_shapes(onnx.load(tmp_path))

    if simplify:
        import onnxsim
        model, check = onnxsim.simplify(model, check_n=1, skip_shape_inference=False)
        print(f"[BERT-Base] simplify valid: {check}, nodes after: {len(model.graph.node)}")

    onnx.save(model, output_path)
    os.remove(tmp_path)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[BERT-Base] exported to {output_path} ({size_mb:.1f} MB)")

    from collections import Counter
    c = Counter(n.op_type for n in model.graph.node)
    print(f"[BERT-Base] op distribution: {dict(c)}")

    return output_path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    os.chdir(repo_root)
    export_bert_base()
