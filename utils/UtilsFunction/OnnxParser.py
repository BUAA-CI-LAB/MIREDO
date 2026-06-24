import os, re, onnx
from onnx import shape_inference
from functools import reduce

# ---------- 工具函数 ---------- #
def natural_key(text):
    """字符块保持字符串；数字块转 int，实现“自然排序”"""
    return [int(t) if t.isdigit() else t
            for t in re.split(r'(\d+)', text)]

def build_shape_dict(onnx_graph):
    """收集 value_info 里所有张量的静态 shape"""
    shape_dict = {}
    def _add(vi):
        dims = [d.dim_value if d.HasField("dim_value") else None
                for d in vi.type.tensor_type.shape.dim]
        shape_dict[vi.name] = dims
    for vi in list(onnx_graph.value_info) + list(onnx_graph.input) + list(onnx_graph.output):
        _add(vi)
    return shape_dict

def loopdims_from_conv_node(node, shape_dict, initializer_dict):
    """提取单个 Conv 节点的 loopDim"""
    W = initializer_dict[node.input[1]]          # 权重：K,Cg,R,S
    Kg, Cg, R, S = W.dims
    group   = next((a.i   for a in node.attribute if a.name == "group"),   1)
    strides = next((a.ints for a in node.attribute if a.name == "strides"), [1, 1])
    pads    = next((a.ints for a in node.attribute if a.name == "pads"),    [0, 0, 0, 0])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    """ 分组卷积的输入输出通道处理，将分组卷积视为 Group 个完全独立的、重复的子任务 """
    """ 将输出通道直接按组切分 不与pytorch的表示方法保持一致 """
    C = Cg
    K = Kg // group
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    X, Y = node.input[0], node.output[0]
    in_shape  = shape_dict.get(X, [None]*4)
    out_shape = shape_dict.get(Y, [None]*4)
    B, H, W0 = in_shape[0],  in_shape[2], in_shape[3]
    _, P, Q  = out_shape[0], out_shape[2], out_shape[3]
    return {
        'R': R, 'S': S, 'C': C, 'K': K,
        'P': P, 'Q': Q, 'G': group, 'B': B,
        'H': H, 'W': W0, 'Stride': strides[0], 'Padding': pads[0]
    }


def _tensor_shape(name, shape_dict, initializer_dict):
    """统一读 tensor 形状：优先 initializer（静态权重），回退到 shape_dict（激活）。"""
    if name in initializer_dict:
        return list(initializer_dict[name].dims)
    shp = shape_dict.get(name)
    if shp is None:
        raise ValueError(f"Cannot resolve shape for tensor {name!r}")
    return list(shp)


def _broadcast_batch(lead_a, lead_b):
    """NumPy-style broadcast leading batch dims, return flattened product for G."""
    if not lead_a and not lead_b:
        return 1
    na, nb = len(lead_a), len(lead_b)
    n = max(na, nb)
    pad_a = [1] * (n - na) + list(lead_a)
    pad_b = [1] * (n - nb) + list(lead_b)
    merged = []
    for a, b in zip(pad_a, pad_b):
        if a == 1:
            merged.append(b if b is not None else 1)
        elif b == 1 or b is None:
            merged.append(a)
        elif a == b:
            merged.append(a)
        else:
            raise ValueError(f"Incompatible batch dims for broadcast: {lead_a} vs {lead_b}")
    return int(reduce(lambda x, y: x * (y or 1), merged, 1))


def loopdims_from_matmul_node(node, shape_dict, initializer_dict):
    """
    提取单个 MatMul 节点的 loopDim，映射为退化卷积：
      R = S = 1, Stride = 1, Padding = 0, H = P, W = 1, Q = 1.
    规则：右操作数 B 作为 W (weight) 角色；左操作数 A 作为 I。
    对 attention 的 Q·K^T 这意味着 K 张量进 crossbar，Q 驱动 word-line；
    对 MLP/Linear 意味着静态权重继续走 W 路径。

    ---- 维度折叠规则 ----
    MatMul 的 leading 维可以是 batch / head 维。语义上需要区分两种情况：
    1. **广播 broadcast**：B 只有 2D（如 MLP 的 weight[K,N]），A 的 leading 维
       意味着"沿着 batch 方向复用同一份权重"。MIREDO 的 G 维假设
       "每组独立权重"（类似 grouped conv），直接放进 G 会把权重 size
       计成 G×（K×N），严重虚增。此时应把 leading 折进 M (P)。
    2. **真正独立实例**（attention 的 Q·K^T）：A 和 B 都有 leading 维，
       每个 (batch, head) 有独立的 Q/K，此时 leading 对应 G (weight 复用
       行为与 grouped conv 一致)。
    """
    A_name, B_name = node.input[0], node.input[1]
    Y_name = node.output[0]

    A_shape = _tensor_shape(A_name, shape_dict, initializer_dict)
    B_shape = _tensor_shape(B_name, shape_dict, initializer_dict)
    Y_shape = _tensor_shape(Y_name, shape_dict, initializer_dict) if Y_name in shape_dict else None

    if len(A_shape) < 2 or len(B_shape) < 2:
        raise ValueError(f"MatMul operands must be rank>=2; got A={A_shape}, B={B_shape}")

    M = int(A_shape[-2])
    K_a = int(A_shape[-1])
    K_b = int(B_shape[-2])
    N = int(B_shape[-1])

    if K_a != K_b:
        raise ValueError(f"MatMul reduction mismatch: A.tail={A_shape[-2:]}, B.tail={B_shape[-2:]}")

    lead_a = A_shape[:-2]
    lead_b = B_shape[:-2]
    # 过滤掉大小 1 的 singleton 维（它们不贡献独立实例数）
    def _effective_lead(shape):
        return [int(d) for d in shape if d is not None and int(d) > 1]

    eff_a = _effective_lead(lead_a)
    eff_b = _effective_lead(lead_b)

    if not eff_b:
        # 右操作数（weight 角色）沿 batch 方向广播复用 → leading 折进 M
        # 这是 MLP / QKV 投影 / FFN 的典型情形
        fold = 1
        for d in eff_a:
            fold *= d
        M_eff = M * fold
        G_eff = 1
    elif not eff_a:
        # 左操作数沿 batch 方向广播（少见，如 score-only 测试图）
        fold = 1
        for d in eff_b:
            fold *= d
        M_eff = M
        # 此时是 weight 独立而 input 复用，保守映射到 G
        G_eff = fold
    else:
        # 双向 leading（attention Q·K^T 和 score·V 的典型情形）
        # G = broadcast(merged leading)；两个 effective lead 必须广播兼容
        G_eff = _broadcast_batch(lead_a, lead_b)
        M_eff = M

    # 一致性校验：Y 形如 leading + [M, N]（若 Y_shape 可用）
    if Y_shape is not None and len(Y_shape) >= 2:
        if Y_shape[-2] not in (M, None) or Y_shape[-1] not in (N, None):
            raise ValueError(f"MatMul output shape {Y_shape} inconsistent with M={M}, N={N}")

    return {
        'R': 1, 'S': 1,
        'C': int(K_a),
        'K': int(N),
        'P': int(M_eff),
        'Q': 1,
        'G': int(G_eff),
        'B': 1,
        'H': int(M_eff),
        'W': 1,
        'Stride': 1,
        'Padding': 0,
    }


def loopdims_from_gemm_node(node, shape_dict, initializer_dict):
    """
    提取单个 Gemm 节点的 loopDim。Gemm: Y = alpha * A * B + beta * C，A/B 均二维，
    支持 transA/transB；C 项不影响 mapping（仅偏置，不计入 MAC 工作量）。
    """
    transA = next((a.i for a in node.attribute if a.name == "transA"), 0)
    transB = next((a.i for a in node.attribute if a.name == "transB"), 0)

    A_name, B_name = node.input[0], node.input[1]
    Y_name = node.output[0]

    A_shape = _tensor_shape(A_name, shape_dict, initializer_dict)
    B_shape = _tensor_shape(B_name, shape_dict, initializer_dict)

    if len(A_shape) != 2 or len(B_shape) != 2:
        raise ValueError(f"Gemm operands must be rank-2; got A={A_shape}, B={B_shape}")

    if transA:
        K_a, M = A_shape
    else:
        M, K_a = A_shape
    if transB:
        N, K_b = B_shape
    else:
        K_b, N = B_shape

    if K_a != K_b:
        raise ValueError(f"Gemm reduction mismatch: A={A_shape} transA={transA}, B={B_shape} transB={transB}")

    return {
        'R': 1, 'S': 1,
        'C': int(K_a),
        'K': int(N),
        'P': int(M),
        'Q': 1,
        'G': 1,
        'B': 1,
        'H': int(M),
        'W': 1,
        'Stride': 1,
        'Padding': 0,
    }

def safe_load_onnx(fp):
    """
    加载 ONNX 模型，但 **不** 去尝试打开 external data 文件；
    只取网络结构与张量形状，足够我们解析 loopDim。
    """
    try:
        # ONNX >=1.12 推荐 load_model，旧版可 fallback 到 load
        if hasattr(onnx, "load_model"):
            model = onnx.load_model(fp, load_external_data=False)
        else:
            model = onnx.load(fp, load_external_data=False)
    except TypeError:
        # 极旧版本没有 load_external_data 参数，退回普通 load
        model = onnx.load(fp)
    return model


# ---------- 单文件解析 ---------- #
_SUPPORTED_OPS = {"Conv", "MatMul", "Gemm"}


def parse_single_onnx(fp: str, allow_matmul: bool = False):
    """返回该 .onnx 文件中所有 Conv/MatMul/Gemm 节点的 layer_names、loopdims。

    默认 allow_matmul=False 保持原 CNN benchmark 的语义（论文明确排除 FC 层）；
    Transformer 路径显式设 allow_matmul=True 以接收 MatMul/Gemm。
    """
    model = shape_inference.infer_shapes(safe_load_onnx(fp))
    g     = model.graph
    shapes = build_shape_dict(g)
    inits  = {i.name: i for i in g.initializer}
    layer_names, loopdims = [], []
    op_counters = {"Conv": 0, "MatMul": 0, "Gemm": 0}
    for node in g.node:
        op_type = node.op_type
        if op_type not in _SUPPORTED_OPS:
            continue
        if op_type != "Conv" and not allow_matmul:
            continue
        try:
            if op_type == "Conv":
                dims = loopdims_from_conv_node(node, shapes, inits)
                lname = f"Conv_{op_counters['Conv']}_{dims['R']}_{dims['S']}_{dims['P']}_" \
                        f"{dims['Q']}_{dims['C']}_{dims['K']}_{dims['G']}"
            elif op_type == "MatMul":
                dims = loopdims_from_matmul_node(node, shapes, inits)
                lname = f"MatMul_{op_counters['MatMul']}_{dims['P']}_{dims['C']}_{dims['K']}_{dims['G']}"
            elif op_type == "Gemm":
                dims = loopdims_from_gemm_node(node, shapes, inits)
                lname = f"Gemm_{op_counters['Gemm']}_{dims['P']}_{dims['C']}_{dims['K']}"
        except Exception as exc:
            # 形状推断失败的节点（动态 shape 等）跳过而非崩溃，保持与 softmax/Add 等
            # silently-skipped 算子一致的处理风格；由上层聚合汇报。
            print(f"[OnnxParser] skip {op_type} node {node.name!r}: {exc}")
            continue
        layer_names.append(lname)
        loopdims.append(dims)
        op_counters[op_type] += 1
    if not layer_names:
        raise RuntimeError(f"{fp} 中未找到 Conv/MatMul/Gemm 节点")
    return layer_names, loopdims

# ---------- 总入口 ---------- #
def extract_loopdims(path: str, allow_matmul: bool = False):
    """
    Parameters
    ----------
    path : str
        .onnx 文件路径 或者 目录路径
    allow_matmul : bool
        True 时同时解析 MatMul / Gemm 节点作为 Transformer matmul/attention 层；
        False（默认）保持原 CNN benchmark 语义（仅 Conv）。

    Returns
    -------
    names : list[str]
        - 文件模式 Conv/MatMul/Gemm 层名字列表
        - 目录模式：每个 .onnx 文件（去掉扩展名）的自然排序列表
    loopdims : list[dict]
        每个名字对应的 loopDim 字典
    """
    if path.lower().endswith(".onnx"):           # ------- 单文件 -------
        return parse_single_onnx(path, allow_matmul=allow_matmul)

    # ---------------------------- 目录 ---------------------------- #
    if not os.path.isdir(path):
        raise FileNotFoundError(path)

    fnames = sorted([f for f in os.listdir(path) if f.endswith(".onnx")],
                    key=natural_key)
    if not fnames:
        raise RuntimeError(f"目录 {path} 中没有 .onnx 文件")

    names, loopdims = [], []
    for f in fnames:
        full = os.path.join(path, f)
        # 假设每个文件只含一层；如有多层可改为取第 0 个
        _, dims_list = parse_single_onnx(full, allow_matmul=allow_matmul)
        names.append(os.path.splitext(f)[0])   # 模型名（去 .onnx）
        loopdims.append(dims_list[0])          # 该文件第一层的数据
    return names, loopdims

# ---------- 示例调用 ---------- #
if __name__ == "__main__":
    p1 = "model/resnet18.onnx"
    p2 = "model/Resnet18"         # 目录

    names1, dims1 = extract_loopdims(p1)
    print("单个模型: ")
    for n, d in zip(names1, dims1):
        print(n)
        print(d)

    names2, dims2 = extract_loopdims(p2)
    print("\n目录: ")
    for n, d in zip(names2, dims2):
        print(n)
        print(d)


