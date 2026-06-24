# MIREDO energy wrapper over the locally vendored CACTI 7.0 (utils/Cacti_wrapper/cacti/).
#
# Exposes two helpers consumed by Architecture/ and Evaluation/common/HardwareVariants.py:
#
#   cacti_power(tech_node, capacity_bytes, bitwidth_bits)
#       -> (r_pJ_per_access, w_pJ_per_access, leak_pJ_per_cycle, leak_mW)
#
#   dram_static(capacity_bytes, bus_width_bits, mem_type='LPDDR4', freq_hz=500e6)
#       -> (leak_mW, leak_pJ_per_cycle)
#
# Both r/w and leakage are derived from the same CACTI invocation so the numbers
# are internally consistent at one technology node (CACTI-supported nodes: 0.022,
# 0.032, 0.040, 0.045, 0.065, 0.090 um; 0.028 remapped to 0.032 + 0.81 scaling).
# This matches the source that originally produced the frozen r_cost_per_bit_pJ
# values in Architecture/templates/default.py.

import hashlib
from pathlib import Path

from utils.Cacti_wrapper.CactiConfig import DEFAULT_CACTI_DIR, get_cacti_cost


def _stable_hd_hash(tech_node: float, capacity_bytes: int, bitwidth_bits: int) -> str:
    key = f"{tech_node}-{int(capacity_bytes)}-{int(bitwidth_bits)}".encode()
    return hashlib.md5(key).hexdigest()[:12]


def cacti_power(tech_node: float,
                capacity_bytes: int,
                bitwidth_bits: int):
    """Run CACTI at `tech_node` (um) for an SRAM of `capacity_bytes`×`bitwidth_bits`.

    Returns (r_pJ_per_access, w_pJ_per_access, leak_pJ_per_cycle, leak_mW). The
    leak_pJ_per_cycle applies the same 28nm/bw>32 scaling as r/w so all four
    outputs live on the same tech-node anchor.
    """
    cap_bytes = max(64, int(capacity_bytes))
    bw = int(bitwidth_bits)
    hd_hash = _stable_hd_hash(tech_node, cap_bytes, bw)

    _access_time_ns, _area_mm2, r_pJ, w_pJ = get_cacti_cost(
        tech_node=tech_node,
        mem_type='sram',
        mem_size_in_byte=cap_bytes,
        bw=bw,
        hd_hash=hd_hash,
    )

    # Re-parse the same CACTI output to extract leakage (get_cacti_cost only
    # returns dynamic energies + access time).
    out_file = Path(DEFAULT_CACTI_DIR) / 'self_gen' / f'cache_{hd_hash}.cfg.out'
    if not out_file.exists():
        raise RuntimeError(f"CACTI output not found: {out_file}")

    with open(out_file) as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise RuntimeError(f"Malformed CACTI output: {out_file}")
    headers = [h.strip() for h in lines[0].split(',')]
    values = [v.strip() for v in lines[-1].split(',')]
    fields = dict(zip(headers, values))

    try:
        leak_mW_raw = float(fields['Standby leakage per bank(mW)'])
        cycle_ns_raw = float(fields['Random cycle time (ns)'])
    except KeyError as e:
        raise RuntimeError(f"Missing field {e} in CACTI output {out_file}")

    # Mirror get_cacti_cost: 28nm → 32nm + (0.9/1.0)^2 = 0.81 scaling; bw>32
    # scaled by bw/32. These keep leakage consistent with the r/w numbers
    # returned alongside (which have already been scaled inside get_cacti_cost).
    scaling_factor = 0.9 * 0.9 if tech_node == 0.028 else 1.0
    bw_scale = bw / 32 if bw > 32 else 1.0
    leak_mW = scaling_factor * leak_mW_raw * bw_scale
    cycle_ns = scaling_factor * cycle_ns_raw
    leak_pJ_per_cycle = leak_mW * cycle_ns  # mW*ns == pJ

    return r_pJ, w_pJ, leak_pJ_per_cycle, leak_mW


def dram_static(capacity_bytes: int,
                bus_width_bits: int = 64,
                mem_type: str = "DDR4",
                freq_hz: float = 500e6):
    """LPDDR4/DDR4/HBM2 static (leakage) power estimation from public datasheet data.

    Parameters
    ----------
    capacity_bytes : int
        DRAM capacity in bytes.
    bus_width_bits : int
        Typical DDR/LPDDR values are 16, 32, 64.
    mem_type : str
        One of {"DDR4", "LPDDR4", "HBM2"}.
    freq_hz : float | None
        Clock frequency (Hz); if provided, also returns leakage energy per cycle.

    Returns
    -------
    (P_leak_mW, E_leak_pJ_per_cycle)
        E_leak_pJ_per_cycle is only meaningful when freq_hz is provided.
    """
    LEAK_PER_GB = {
        "DDR4":   42.0,  # 8 Gb x16 DDR4: IDD2N=35 mA @1.2V → ~42 mW
        "LPDDR4":  9.0,  # 8 Gb LPDDR4: IDD2N≈8 mA @1.1V   → ~8.8 mW
        "HBM2":   20.0,  # Typical public Samsung/AMD value
    }
    if mem_type not in LEAK_PER_GB:
        raise ValueError(f"mem_type must be one of {list(LEAK_PER_GB)}")

    cap_gib = capacity_bytes / 2**30
    width_fac = bus_width_bits / 64

    P_leak_mW = LEAK_PER_GB[mem_type] * cap_gib * width_fac
    if freq_hz is None:
        return P_leak_mW
    E_leak_pJ = P_leak_mW * 1e-3 / freq_hz * 1e12
    return P_leak_mW, E_leak_pJ


if __name__ == "__main__":
    # Match the Global_buffer configuration from Architecture/templates/default.py
    r, w, leak_pJ, leak_mW = cacti_power(tech_node=0.028,
                                          capacity_bytes=2097152 // 8,
                                          bitwidth_bits=128)
    print(f"Global_buffer @28nm:")
    print(f"  r_cost  = {r/128:.6f} pJ/bit  (expect 0.197874)")
    print(f"  w_cost  = {w/128:.6f} pJ/bit  (expect 0.142806)")
    print(f"  leak    = {leak_pJ:.6f} pJ/cycle   leak_mW = {leak_mW:.6f}")
