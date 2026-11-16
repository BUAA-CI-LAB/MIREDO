import os
import onnx
import torch
from onnx import shape_inference


def export_bert_base(output_path: str = "model/bert_base.onnx",
                     seq_len: int = 128,
                     batch_size: int = 1,
                     opset_version: int = 13,
                     simplify: bool = True) -> str:
    from transformers import BertConfig, BertModel
    cfg = BertConfig()
    m = BertModel(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    tmp_path = output_path + ".tmp"
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
