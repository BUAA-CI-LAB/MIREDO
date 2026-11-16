import os, re, onnx
from onnx import shape_inference
from functools import reduce

def natural_key(text):
    return [int(t) if t.isdigit() else t
            for t in re.split(r'(\d+)', text)]

def build_shape_dict(onnx_graph):
    shape_dict = {}
    def _add(vi):
        dims = [d.dim_value if d.HasField("dim_value") else None
                for d in vi.type.tensor_type.shape.dim]
        shape_dict[vi.name] = dims
    for vi in list(onnx_graph.value_info) + list(onnx_graph.input) + list(onnx_graph.output):
        _add(vi)
    return shape_dict

def loopdims_from_conv_node(node, shape_dict, initializer_dict):
    W = initializer_dict[node.input[1]]
    Kg, Cg, R, S = W.dims
    group   = next((a.i   for a in node.attribute if a.name == "group"),   1)
    strides = next((a.ints for a in node.attribute if a.name == "strides"), [1, 1])
    pads    = next((a.ints for a in node.attribute if a.name == "pads"),    [0, 0, 0, 0])
    C = Cg
    K = Kg // group
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
    if name in initializer_dict:
        return list(initializer_dict[name].dims)
    shp = shape_dict.get(name)
    if shp is None:
        raise ValueError(f"Cannot resolve shape for tensor {name!r}")
    return list(shp)


def _broadcast_batch(lead_a, lead_b):
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
    def _effective_lead(shape):
        return [int(d) for d in shape if d is not None and int(d) > 1]

    eff_a = _effective_lead(lead_a)
    eff_b = _effective_lead(lead_b)

    if not eff_b:
        fold = 1
        for d in eff_a:
            fold *= d
        M_eff = M * fold
        G_eff = 1
    elif not eff_a:
        fold = 1
        for d in eff_b:
            fold *= d
        M_eff = M
        G_eff = fold
    else:
        G_eff = _broadcast_batch(lead_a, lead_b)
        M_eff = M

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
    try:
        if hasattr(onnx, "load_model"):
            model = onnx.load_model(fp, load_external_data=False)
        else:
            model = onnx.load(fp, load_external_data=False)
    except TypeError:
        model = onnx.load(fp)
    return model


_SUPPORTED_OPS = {"Conv", "MatMul", "Gemm"}


def parse_single_onnx(fp: str, allow_matmul: bool = False):
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
            print(f"[OnnxParser] skip {op_type} node {node.name!r}: {exc}")
            continue
        layer_names.append(lname)
        loopdims.append(dims)
        op_counters[op_type] += 1
    if not layer_names:
        raise RuntimeError(f"{fp} 中未找到 Conv/MatMul/Gemm 节点")
    return layer_names, loopdims

def extract_loopdims(path: str, allow_matmul: bool = False):
    if path.lower().endswith(".onnx"):
        return parse_single_onnx(path, allow_matmul=allow_matmul)

    if not os.path.isdir(path):
        raise FileNotFoundError(path)

    fnames = sorted([f for f in os.listdir(path) if f.endswith(".onnx")],
                    key=natural_key)
    if not fnames:
        raise RuntimeError(f"目录 {path} 中没有 .onnx 文件")

    names, loopdims = [], []
    for f in fnames:
        full = os.path.join(path, f)
        _, dims_list = parse_single_onnx(full, allow_matmul=allow_matmul)
        names.append(os.path.splitext(f)[0])
        loopdims.append(dims_list[0])
    return names, loopdims

if __name__ == "__main__":
    p1 = "model/resnet18.onnx"
    p2 = "model/Resnet18"

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
