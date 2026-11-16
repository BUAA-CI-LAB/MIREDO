from pathlib import Path
import sys

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def zigzag_submodule_root() -> Path:
    return repo_root() / "Evaluation" / "Zigzag_imc" / "zigzag-imc"


def ensure_zigzag_submodule_on_path() -> Path:
    root = zigzag_submodule_root()
    api_file = root / "zigzag" / "api.py"
    if not api_file.is_file():
        raise FileNotFoundError(f"ZigZag submodule is missing: {api_file}")

    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    return root


def convert_Zigzag_to_MIREDO(*args, **kwargs):
    from Evaluation.Zigzag_imc.CompatibleZigzag import convert_Zigzag_to_MIREDO as compat_api
    return compat_api(*args, **kwargs)
