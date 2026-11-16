from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path

from Evaluation.common.EvalCommon import OBJECTIVES


SUPPORTED_BASELINE_METHODS = ("ws", "zigzag", "cimloop")
BASELINE_METHOD_LABELS = {"ws": "WS", "zigzag": "ZigZag-IMC", "cimloop": "CiMLoop"}
_ADAPTERS = {
    "ws": "Evaluation.WeightStationaryGenerator",
    "zigzag": "Evaluation.Zigzag_imc.zigzag_adapter",
    "cimloop": "Evaluation.CIMLoop.cimloop_adapter",
}


@dataclass
class BaselineRunResult:
    method: str
    objective: str
    latency: float
    energy: float
    profile: object
    dataflow: object
    metadata: dict = field(default_factory=dict)

    @property
    def edp(self):
        return self.latency * self.energy


def run_baseline(method, acc, ops, loopdim, model_name, architecture,
                 objective="EDP", *, work_dir=None, loma_lpf_limit=6, mapper_threads=4):
    method = method.lower()
    if method not in _ADAPTERS:
        raise ValueError(f"Unknown baseline {method!r}; choose {SUPPORTED_BASELINE_METHODS}")
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective {objective!r}; choose {OBJECTIVES}")
    if loma_lpf_limit < 1 or mapper_threads < 1:
        raise ValueError("Search limits and thread counts must be positive")
    adapter = import_module(_ADAPTERS[method])
    reason = adapter.supports_loopdim(loopdim)
    if reason:
        raise ValueError(f"{method} does not support this layer: {reason}")
    kwargs = dict(acc=acc, ops=ops, loopdim=loopdim, model_name=model_name,
                  architecture=architecture, objective=objective)
    if method != "ws":
        if work_dir is None:
            raise ValueError("Mapper baselines require an output directory")
        kwargs["work_dir"] = Path(work_dir).resolve()
    if method == "zigzag":
        kwargs["loma_lpf_limit"] = loma_lpf_limit
    elif method == "cimloop":
        kwargs["mapper_options"] = {"mapper_num_threads": mapper_threads}
    return adapter.run_for_layer(**kwargs)
