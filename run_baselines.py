import argparse
import copy
import json
import math
import re
from pathlib import Path
from uuid import uuid4

from Evaluation.common.BaselineProvider import (
    BASELINE_METHOD_LABELS, SUPPORTED_BASELINE_METHODS, run_baseline,
)
from Evaluation.common.EvalCommon import ARCHITECTURES, OBJECTIVES, make_accelerator, repo_root


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def arguments(argv=None):
    parser = argparse.ArgumentParser(description='Run one baseline and evaluate its mappings on the common simulator.')
    parser.add_argument("--baseline", required=True, choices=SUPPORTED_BASELINE_METHODS)
    parser.add_argument("-m", "--model", required=True,
                        help="ONNX path or a name under model/")
    parser.add_argument("-a", "--architecture", choices=ARCHITECTURES, default="default")
    parser.add_argument("--matmul", action="store_true", help="include MatMul/Gemm operators")
    parser.add_argument("-opt", "--objective", choices=OBJECTIVES, default="EDP")
    parser.add_argument("--layer", type=positive_int, help="evaluate one layer (1-based index)")
    parser.add_argument("--lpf-limit", type=positive_int, default=6,
                        help="ZigZag temporal-factor search limit (default: 6)")
    parser.add_argument("--mapper-threads", type=positive_int, default=4,
                        help="CiMLoop mapper threads (default: 4); layers run sequentially")
    parser.add_argument("-o", "--outputdir", type=Path)
    return parser.parse_args(argv)


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main(argv=None):
    args = arguments(argv)
    from utils.GlobalUT import CONST
    from utils.Workload import WorkLoad
    from utils.UtilsFunction.OnnxParser import extract_loopdims

    model = Path(args.model)
    if model.suffix.lower() != ".onnx":
        model = repo_root() / "model" / f"{args.model}.onnx"
    model = model.resolve()
    if not model.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model}")
    names, dimensions = extract_loopdims(str(model), allow_matmul=args.matmul)
    if not names or len(names) != len(dimensions):
        raise ValueError("The model contains no supported layers")
    indices = range(len(names)) if args.layer is None else [args.layer - 1]
    if args.layer is not None and args.layer > len(names):
        raise ValueError(f"Layer index exceeds the {len(names)} supported layers")

    output = args.outputdir or repo_root() / "output" / f"{args.baseline}_{model.stem}_{uuid4().hex[:8]}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists():
        raise FileExistsError(f"Output already contains results: {result_path}")
    CONST.FLAG_OPT = args.objective
    template = make_accelerator(args.architecture)
    records = []
    for i in indices:
        loopdim = copy.deepcopy(dimensions[i])
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", names[i])[:180]
        work_dir = output / f"{i + 1:03d}_{safe_name}"
        work_dir.mkdir()
        result = run_baseline(
            method=args.baseline, acc=copy.deepcopy(template), ops=WorkLoad(loopDim=loopdim),
            loopdim=loopdim, model_name=model.stem, architecture=args.architecture,
            objective=args.objective, work_dir=work_dir,
            loma_lpf_limit=args.lpf_limit, mapper_threads=args.mapper_threads,
        )
        if not math.isfinite(result.latency) or not math.isfinite(result.energy):
            raise ValueError(f"Non-finite result for {names[i]}")
        records.append({"layer": names[i], "index": i + 1, "loopdim": loopdim,
                        "latency_cycles": float(result.latency), "energy_nJ": float(result.energy),
                        "edp": float(result.edp), "metadata": result.metadata})
        print(f"{names[i]}: {result.latency:g} cycles, {result.energy:g} nJ", flush=True)

    latency = math.fsum(r["latency_cycles"] for r in records)
    energy = math.fsum(r["energy_nJ"] for r in records)
    payload = {
        "baseline": BASELINE_METHOD_LABELS[args.baseline], "model": str(model),
        "objective": args.objective, "architecture": args.architecture,
        "hardware": template.source_spec.to_dict(),
        "search": {"loma_lpf_limit": args.lpf_limit, "mapper_threads": args.mapper_threads},
        "per_layer": records,
        "total": {"layers": len(records), "latency_cycles": latency, "energy_nJ": energy,
                  "edp": latency * energy},
        "edp_definition": "sum(layer latency) * sum(layer energy)",
    }
    result_path.write_text(json.dumps(payload, indent=2, default=json_value, allow_nan=False) + "\n")
    print(result_path)


if __name__ == "__main__":
    main()
