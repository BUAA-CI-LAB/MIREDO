# HW-Transformer 配置 —— 中档移动级 Transformer CIM 加速器
#
# 从 HW-Small (templates/default.py) 派生，参数放大以容纳 ViT-Base/16 (seq=197, d_model=768,
# MLP hidden=3072) 和 BERT-Base (seq=128, d_model=768) 而不强制退化到 DRAM-bound mapping。
#
# 参数锚点：HAMMER (HPCA 2023, 28 nm, 16 TOPS / 1.5 MB SRAM, 32×{256×256} 宏)、
# SpAtten (HPCA 2021)、Sanger (MICRO 2021)、Samsung HBM-PIM (ISSCC 2022)、
# 近年 28 nm SRAM-CIM ISSCC 设计。
#
# 计算：16 cores × (64×32) bit-serial macro → 32768 bit-level MACs/cycle
#       ÷ 8 bit-serial cycles = 4096 INT8×INT8 MACs/cycle @ 1 GHz → 32.77 TOPS
# 片上 SRAM：4 MB GBuf + 16×(16 KB + 16 KB) = 4.5 MB 总容量
# DRAM：4 GB @ 256 bit/cycle (LPDDR5x 级，匹配 32 TOPS 计算需求)
#
# 能耗/漏电字段由 transformer_spec() 在返回前统一通过 CACTI 重跑，**不要手工填 pJ 值**。

from __future__ import annotations

import copy
from dataclasses import replace

from Architecture.HardwareSpec import HardwareSpec
from Architecture.templates.default import _DEFAULT_SPEC_DICT


def _build_transformer_spec_dict() -> dict:
    spec = copy.deepcopy(_DEFAULT_SPEC_DICT)

    # === 计算轴 ===
    spec["cores"] = 16
    spec["macro"]["dimX"] = 64
    spec["macro"]["dimY"] = 32
    for axis in spec["macro"]["spatial_axes"]:
        if axis["name"] == "cores":
            axis["size"] = 16
        elif axis["name"] == "dimX":
            axis["size"] = 64
        elif axis["name"] == "dimY":
            axis["size"] = 32

    # === 存储层级 ===
    # DRAM: 4 GB @ 256 bit/cycle
    # Global_buffer: 4 MB @ 512 bit/cycle
    # I/O buffer (per core): 16 KB @ 256 bit/cycle
    # OReg/IReg/Macro size_bits 保持原值（per-cell 位宽语义，不随 macro 维度换算）
    _SIZE_OVERRIDES = {
        "Dram":           {"size_bits": 4 * 8 * (1024 ** 3), "r_bw_bits_per_cycle": 256, "w_bw_bits_per_cycle": 256},
        "Global_buffer":  {"size_bits": 4 * 8 * (1024 ** 2), "r_bw_bits_per_cycle": 512, "w_bw_bits_per_cycle": 512},
        "Output_buffer":  {"size_bits": 16 * 8 * 1024,        "r_bw_bits_per_cycle": 256, "w_bw_bits_per_cycle": 256},
        "Input_buffer":   {"size_bits": 16 * 8 * 1024,        "r_bw_bits_per_cycle": 256, "w_bw_bits_per_cycle": 256},
    }
    for mem in spec["memory_hierarchy"]:
        if mem["name"] in _SIZE_OVERRIDES:
            mem.update(_SIZE_OVERRIDES[mem["name"]])

    # 元数据
    spec["metadata"] = dict(spec["metadata"])
    spec["metadata"]["notes"] = (
        "HW-Transformer: 16-core digital SRAM CIM, 28nm, I=W=8b psum=16b, "
        "scaled for ViT/BERT-Base (16 cores, 64x32 macro, 4 MB GBuf, 16 KB IO buffer, 4 GB DRAM)"
    )

    return spec


def transformer_spec() -> HardwareSpec:
    """返回 HW-Transformer 配置的 HardwareSpec；所有 SRAM 能耗/漏电走 CACTI。"""
    # 延迟导入避免循环依赖 (HardwareVariants 自身导入 HardwareSpec / CACTI wrapper)
    from Architecture.HardwareVariants import (
        _recompute_memory_cost_pJ,
        _leakage_per_cycle_nJ,
    )

    spec = HardwareSpec.from_dict(_build_transformer_spec_dict())

    # DRAM 保持 7.91 pJ/bit legacy；OReg/IReg/Macro 走 spec 里的原始固定值
    # （OReg/IReg 静态能耗=0 符合 flop-level 寄存器语义）。
    # 对 Global_buffer / Input_buffer / Output_buffer 三个 SRAM 层，重跑 CACTI。
    cacti_levels = ("Global_buffer", "Input_buffer", "Output_buffer")
    refreshed_mems = []
    for m in spec.memory_hierarchy:
        if m.name in cacti_levels:
            updated = _recompute_memory_cost_pJ(spec, m.name)
            refreshed_mems.append(replace(
                m,
                r_cost_per_bit_pJ=updated.r_cost_per_bit_pJ,
                w_cost_per_bit_pJ=updated.w_cost_per_bit_pJ,
            ))
        else:
            refreshed_mems.append(m)
    spec = replace(spec, memory_hierarchy=refreshed_mems)

    # 总漏电按新 spec 累加 (DRAM + GBuf + per-core SRAM × cores + Macro × cores)
    spec = replace(spec, leakage_per_cycle_nJ=_leakage_per_cycle_nJ(spec))

    return spec


default_spec = transformer_spec
