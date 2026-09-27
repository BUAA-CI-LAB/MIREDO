"""Single-token decode snapshots of the GPT-2-medium and TinyLlama-1.1B blocks.

Exports model/gpt2_medium_decode_b{B}_t{T}.onnx and
model/tinyllama_decode_b{B}_t{T}.onnx (shape-inferred + onnx-simplifier).

Each file is one decoder block executing ONE new token per sequence for a batch
of B independent sequences at context length T (T includes the new token, so
the KV cache holds T-1 past entries). The block re-uses the prefill module
classes from GPT2_Medium_block.py / TinyLlama_block.py unchanged; only the
attention operands differ:

    Q                 : (B, heads, 1, d)
    K, V after concat : (B, kv_heads -> heads, T, d)   (TinyLlama repeats K/V
                                                        from 4 to 32 heads as in
                                                        the prefill export)
    QK^T              : (B, heads, 1, d) x (B, heads, d, T) -> (B, heads, 1, T)
    Score.V           : (B, heads, 1, T) x (B, heads, T, d) -> (B, heads, 1, d)

Matrix inventory seen by OnnxParser (Conv/MatMul/Gemm only):
    GPT-2-medium : 6 linear (Q, K, V, O, FC1, FC2) with M = B, + QK^T + Score.V
    TinyLlama    : 7 linear (Q, K, V, O, gate, up, down) with M = B, + QK^T + Score.V

Execution assumptions follow MIREDO's per-layer abstraction: every
matrix product is an independent cold-start operator; projection weights and the
KV cache are streamed from DRAM for every token; no cross-token residency, KV
append, fusion or cross-operator scheduling is modeled. Softmax, normalization,
activation, residual, RoPE, Concat and Reshape/Transpose are dropped by the
parser exactly as in the prefill exports. Random weights; shapes only.

Usage:
    python model/Decode_blocks.py \
        --batch 1 4 16 --context 128 512 1024 --grid cross
    --grid cross  : b=1 x every context, plus every batch x the last context
    --grid full   : every (batch, context) pair
"""

import argparse
import os
import sys

import onnx
import torch
import torch.nn as nn
from onnx import shape_inference

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from GPT2_Medium_block import GPT2MediumBlock  # noqa: E402
from TinyLlama_block import TinyLlamaBlock  # noqa: E402


class DecodeStep(nn.Module):
    """Wrap a prefill block module and run one token against a KV cache."""

    def __init__(self, block: nn.Module, kv_repeat: int = 1):
        super().__init__()
        self.block = block
        self.kv_repeat = kv_repeat

    def forward(self, x, past_k, past_v):
        blk = self.block
        bsz, seq, _ = x.shape          # seq == 1
        n_heads, d = blk.num_attention_heads, blk.head_dim
        n_kv = getattr(blk, "num_key_value_heads", n_heads)

        residual = x
        q = blk.q_proj(x).view(bsz, seq, n_heads, d).transpose(1, 2)
        k = blk.k_proj(x).view(bsz, seq, n_kv, d).transpose(1, 2)
        v = blk.v_proj(x).view(bsz, seq, n_kv, d).transpose(1, 2)
        k = torch.cat([past_k, k], dim=2)   # (B, n_kv, T, d)
        v = torch.cat([past_v, v], dim=2)
        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (d ** -0.5)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, seq, n_heads * d)
        x = residual + blk.o_proj(ctx)

        residual = x
        if hasattr(blk, "gate_proj"):
            gate = blk.gate_proj(x)
            up = blk.up_proj(x)
            x = residual + blk.down_proj(torch.nn.functional.silu(gate) * up)
        else:
            h = blk.fc1(x)
            h = torch.nn.functional.gelu(h)
            x = residual + blk.fc2(h)
        return x


MODELS = {
    "gpt2_medium_decode": dict(cls=GPT2MediumBlock, hidden=1024, heads=16, kv_heads=16, d=64),
    "tinyllama_decode": dict(cls=TinyLlamaBlock, hidden=2048, heads=32, kv_heads=4, d=64),
}


def export_decode(name, batch, context, opset_version=13, simplify=True):
    spec = MODELS[name]
    block = spec["cls"]().eval()
    step = DecodeStep(block, kv_repeat=spec["heads"] // spec["kv_heads"]).eval()

    x = torch.randn(batch, 1, spec["hidden"])
    past_k = torch.randn(batch, spec["kv_heads"], context - 1, spec["d"])
    past_v = torch.randn(batch, spec["kv_heads"], context - 1, spec["d"])

    output_path = f"model/{name}_b{batch}_t{context}.onnx"
    tmp_path = output_path + ".tmp"
    torch.onnx.export(
        step, (x, past_k, past_v), tmp_path, opset_version=opset_version,
        input_names=["hidden_states", "past_key", "past_value"],
        output_names=["block_output"], do_constant_folding=True,
    )
    model = shape_inference.infer_shapes(onnx.load(tmp_path))
    if simplify:
        import onnxsim
        model, check = onnxsim.simplify(model, check_n=1, skip_shape_inference=False)
        print(f"[{name} b{batch} t{context}] simplify valid: {check}, nodes after: {len(model.graph.node)}")
    onnx.save(model, output_path)
    os.remove(tmp_path)

    from collections import Counter
    c = Counter(n.op_type for n in model.graph.node)
    print(f"[{name} b{batch} t{context}] exported to {output_path}; op distribution: {dict(c)}")
    return output_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--batch", nargs="+", type=int, default=[1, 4, 16])
    p.add_argument("--context", nargs="+", type=int, default=[128, 512, 1024])
    p.add_argument("--grid", choices=["cross", "full"], default="cross")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(here))

    if args.grid == "full":
        pairs = [(b, t) for b in args.batch for t in args.context]
    else:
        pairs = [(args.batch[0], t) for t in args.context]
        pairs += [(b, args.context[-1]) for b in args.batch[1:]]
    for name in args.models:
        for b, t in pairs:
            export_decode(name, b, t)


if __name__ == "__main__":
    main()
