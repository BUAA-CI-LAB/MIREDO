import os
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import shape_inference


class GPT2MediumBlock(nn.Module):

    def __init__(self,
                 hidden_size: int = 1024,
                 intermediate_size: int = 4096,
                 num_attention_heads: int = 16,
                 head_dim: int = 64):
        super().__init__()
        assert num_attention_heads * head_dim == hidden_size, \
            f"heads*head_dim={num_attention_heads*head_dim} != hidden={hidden_size}"
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=True)

        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        residual = x

        q = self.q_proj(x).view(bsz, seq, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, self.num_attention_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, seq, self.num_attention_heads * self.head_dim)
        x = residual + self.o_proj(ctx)

        residual = x
        h = self.fc1(x)
        h = F.gelu(h)
        h = self.fc2(h)
        x = residual + h
        return x


def export_gpt2_medium_block(output_path: str = "model/gpt2_medium_block.onnx",
                             seq_len: int = 1024,
                             batch_size: int = 1,
                             opset_version: int = 13,
                             simplify: bool = True) -> str:
    block = GPT2MediumBlock().eval()

    x = torch.randn(batch_size, seq_len, 1024, dtype=torch.float32)

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
        print(f"[GPT-2-medium] simplify valid: {check}, nodes after: {len(model.graph.node)}")

    onnx.save(model, output_path)
    os.remove(tmp_path)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[GPT-2-medium] exported to {output_path} ({size_mb:.1f} MB)")

    from collections import Counter
    c = Counter(n.op_type for n in model.graph.node)
    print(f"[GPT-2-medium] op distribution: {dict(c)}")

    return output_path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    os.chdir(repo_root)
    export_gpt2_medium_block()
