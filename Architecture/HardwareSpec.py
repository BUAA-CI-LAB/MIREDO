# MIREDO HardwareSpec —— CIM accelerator 物理硬件的唯一数据源
#
# 描述 digital SRAM CIM 加速器。由 CIM_Acc.from_spec() 构造 MIREDO
# 内部表示；由 zigzag_adapter.to_zigzag_accelerator() 翻译到 ZigZag。
# MIREDO 内部索引表（mem2dict、nxtMem 等）不在此存储，由构造器派生。

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Union


@dataclass
class PrecisionSpec:
    I: int
    W: int
    psum: int
    O_final: int

    @classmethod
    def from_dict(cls, d: dict) -> "PrecisionSpec":
        return cls(**d)


@dataclass
class LogicEnergies:
    mult_1b: float
    adder_1b: float
    reg_1b: float

    @classmethod
    def from_dict(cls, d: dict) -> "LogicEnergies":
        return cls(**d)


@dataclass
class TechParams:
    tech_node: float
    vdd: float
    nd2_cap: float
    xor2_cap: float
    dff_cap: float
    nd2_area: float
    xor2_area: float
    dff_area: float
    nd2_dly: float
    xor2_dly: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TechParams":
        return cls(**d)


@dataclass
class SpatialAxisSpec:
    name: str
    size: int
    allowed_loops: List[str]
    source_memory_per_operand: Dict[str, str]

    @classmethod
    def from_dict(cls, d: dict) -> "SpatialAxisSpec":
        return cls(
            name=d["name"],
            size=d["size"],
            allowed_loops=list(d["allowed_loops"]),
            source_memory_per_operand=dict(d["source_memory_per_operand"]),
        )


@dataclass
class MacroSpec:
    dimX: int
    dimY: int
    input_bit_per_cycle: int
    precision: PrecisionSpec
    logic_energies_pJ: LogicEnergies
    tech_params: TechParams
    spatial_axes: List[SpatialAxisSpec]
    compartment_depth: int = 8

    @classmethod
    def from_dict(cls, d: dict) -> "MacroSpec":
        return cls(
            dimX=d["dimX"],
            dimY=d["dimY"],
            input_bit_per_cycle=d["input_bit_per_cycle"],
            precision=PrecisionSpec.from_dict(d["precision"]),
            logic_energies_pJ=LogicEnergies.from_dict(d["logic_energies_pJ"]),
            tech_params=TechParams.from_dict(d["tech_params"]),
            spatial_axes=[SpatialAxisSpec.from_dict(a) for a in d["spatial_axes"]],
            compartment_depth=d.get("compartment_depth", 8),
        )


@dataclass
class PortConfig:
    r: int = 0
    w: int = 0
    rw: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "PortConfig":
        return cls(**d)


@dataclass
class MemoryLevelSpec:
    name: str
    size_bits: int
    replication: Literal["per_core", "shared_all_cores"]
    r_bw_bits_per_cycle: int
    w_bw_bits_per_cycle: int
    r_cost_per_bit_pJ: float
    w_cost_per_bit_pJ: float
    operands: List[str]
    served_dimensions_zigzag: Union[str, List[List[int]]]
    area_mm2: float = 0.0
    r_latency_cycles: int = 1
    w_latency_cycles: int = 1
    ports: PortConfig = field(default_factory=PortConfig)
    min_r_granularity_bits: Optional[int] = None
    min_w_granularity_bits: Optional[int] = None

    @property
    def shared(self) -> bool:
        return len(self.operands) > 1

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryLevelSpec":
        kwargs: Dict[str, Any] = {
            k: v for k, v in d.items() if k in cls.__dataclass_fields__
        }
        kwargs["operands"] = list(kwargs["operands"])
        kwargs["ports"] = PortConfig.from_dict(kwargs["ports"])
        sd = kwargs["served_dimensions_zigzag"]
        if isinstance(sd, list):
            kwargs["served_dimensions_zigzag"] = [list(t) for t in sd]
        return cls(**kwargs)


@dataclass
class MetadataSpec:
    tech_node: str = "28nm"
    imc_family: str = "digital_SRAM_IMC"
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "MetadataSpec":
        return cls(**d)


@dataclass
class HardwareSpec:
    cores: int
    cycle_time_ns: Optional[float]
    leakage_per_cycle_nJ: float
    macro: MacroSpec
    memory_hierarchy: List[MemoryLevelSpec]
    metadata: MetadataSpec = field(default_factory=MetadataSpec)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HardwareSpec":
        return cls(
            cores=d["cores"],
            cycle_time_ns=d.get("cycle_time_ns"),
            leakage_per_cycle_nJ=d["leakage_per_cycle_nJ"],
            macro=MacroSpec.from_dict(d["macro"]),
            memory_hierarchy=[MemoryLevelSpec.from_dict(m) for m in d["memory_hierarchy"]],
            metadata=MetadataSpec.from_dict(d.get("metadata", {})),
        )

    def memory_by_name(self, name: str) -> MemoryLevelSpec:
        for m in self.memory_hierarchy:
            if m.name == name:
                return m
        raise KeyError(f"Memory level {name!r} not in spec")
