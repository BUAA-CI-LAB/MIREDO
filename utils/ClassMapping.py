from dataclasses import dataclass
from typing import Any


@dataclass
class ClassMapping:
    source: str
    loop_dim: dict[str, int]
    temporal_mapping: dict[str, list[list[tuple[str, int]]]]
    spatial_mapping: dict[str, list[list[tuple[str, int]]]]
    double_buffer_flag: dict[str, list[bool]]
    top_r_loop_size: dict[str, list[int]] | None = None
    raw: Any = None
