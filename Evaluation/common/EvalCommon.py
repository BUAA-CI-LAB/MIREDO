from importlib import import_module
from pathlib import Path


ARCHITECTURES = ("default", "transformer")
OBJECTIVES = ("Latency", "Energy", "EDP")


def repo_root():
    return Path(__file__).resolve().parents[2]


def objective_to_opt_flag(objective):
    return {"Latency": "latency", "Energy": "energy", "EDP": "EDP"}[objective]


def hardware_spec(architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}; choose {ARCHITECTURES}")
    return import_module(f"Architecture.templates.{architecture}").default_spec()


def resolve_spec(acc, architecture):
    spec = getattr(acc, "source_spec", None)
    return spec if spec is not None else hardware_spec(architecture)


def make_accelerator(architecture):
    from Architecture.ArchSpec import CIM_Acc
    return CIM_Acc.from_spec(hardware_spec(architecture))
