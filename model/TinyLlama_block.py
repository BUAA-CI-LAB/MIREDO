"""TinyLlama-1.1B prefill transformer block ONNX export.

Generates model/tinyllama_block.onnx (shape-inferred + onnx-simplifier).
Exports a single LLaMA-2-architecture decoder block at TinyLlama-1.1B scale,
with Grouped-Query Attention (GQA, 4 KV heads), random weights — MIREDO only
consumes the operator shapes.

TinyLlama-1.1B reference architecture (TinyLlama/TinyLlama-1.1B-Chat-v1.0):
    hidden_size            = 2048
    intermediate_size      = 5632
    num_attention_heads    = 32
    num_key_value_heads    = 4       # GQA
    head_dim               = 64       (2048/32)
    Default prefill seq    = 1024 (chosen to match GPT-2-medium pillar; the
                                   model itself was trained at seq=2048 but
                                   the shape-only ONNX export is sequence-agnostic).

GEMM inventory the ONNX file exposes (the parser keeps Conv/MatMul/Gemm only):
    Linear (Gemm) × 7   : Q, K, V, O, gate, up, down projections
    MatMul × 2          : QK^T, Score·V (after KV repeat from 4 to 32 heads)

Usage:
    /home/xiaolin/anaconda3/envs/pim/bin/python model/TinyLlama_block.py
"""

import os
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import shape_inference


class TinyLlamaBlock(nn.Module):
    """Single TinyLlama-1.1B decoder block with GQA, no KV-cache (prefill)."""

    def __init__(self,
                 hidden_size: int = 2048,
                 intermediate_size: int = 5632,
                 num_attention_heads: int = 32,
                 num_key_value_heads: int = 4,
                 head_dim: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.kv_repeat = num_attention_heads // num_key_value_heads
        assert num_attention_heads % num_key_value_heads == 0

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        residual = x

        q = self.q_proj(x).view(bsz, seq, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.kv_repeat, dim=1)
        v = v.repeat_interleave(self.kv_repeat, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, seq, self.num_attention_heads * self.head_dim)
        x = residual + self.o_proj(ctx)

        residual = x
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = residual + self.down_proj(F.silu(gate) * up)
        return x


def export_tinyllama_block(output_path: str = "model/tinyllama_block.onnx",
                           seq_len: int = 1024,
                           batch_size: int = 1,
                           opset_version: int = 13,
                           simplify: bool = True) -> str:
    """Export single TinyLlama-1.1B decoder block to ONNX (random weights)."""
    block = TinyLlamaBlock().eval()

    x = torch.randn(batch_size, seq_len, 2048, dtype=torch.float32)

    tmp_path = output_path + ".tmp"
    torch.onnx.export(
        block,
        (x,),
        tmp_path,
        opset_version=opset_version,
        input_names=["hidden_states"],
        output_names=["block_output"],
        do_constant_folding=True,
    )

    model = shape_inference.infer_shapes(onnx.load(tmp_path))

    if simplify:
        import onnxsim
        model, check = onnxsim.simplify(model, check_n=1, skip_shape_inference=False)
        print(f"[TinyLlama-1.1B] simplify valid: {check}, nodes after: {len(model.graph.node)}")

    onnx.save(model, output_path)
    os.remove(tmp_path)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[TinyLlama-1.1B] exported to {output_path} ({size_mb:.1f} MB)")

    from collections import Counter
    c = Counter(n.op_type for n in model.graph.node)
    print(f"[TinyLlama-1.1B] op distribution: {dict(c)}")

    return output_path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    os.chdir(repo_root)
    export_tinyllama_block()
