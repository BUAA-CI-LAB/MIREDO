# this file is prepared for project 511
# Created by iboxl

from functools import lru_cache
import math
from typing import List, Tuple, Optional
from itertools import combinations

# ----------------------------------------------------------------------
# 1) 常用工具
# ----------------------------------------------------------------------
def prime_factors(n: int) -> List[int]:
    """返回 n 的素因子（含重复），升序。"""
    if n <= 1:
        raise ValueError("N 必须 > 1")
    factors, p = [], 2
    while p * p <= n:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 1 if p == 2 else 2
    if n > 1:
        factors.append(n)
    return factors                      # 升序

# Not use
@lru_cache(maxsize=None)
def divisors(x: int) -> Tuple[int, ...]:
    ds = []
    for i in range(1, math.isqrt(x) + 1):
        if x % i == 0:
            ds.append(i)
            if i != x // i:
                ds.append(x // i)
    return tuple(sorted(ds))

# Not use
def reachable_products(factors: Tuple[int, ...]) -> int:
    """返回可达循环尺寸个数（去 1）"""
    n, prods = len(factors), set()
    def dfs(idx: int, groups: List[List[int]]):
        if idx == n:
            for g in groups:
                p = 1
                for v in g:
                    p *= v
                if p != 1:
                    prods.add(p)
            return
        # 放入已有分组
        for g in groups:
            g.append(factors[idx])
            dfs(idx + 1, groups)
            g.pop()
        # 自成一组
        groups.append([factors[idx]])
        dfs(idx + 1, groups)
        groups.pop()
    dfs(0, [])
    return len(prods)

# Not use
# ----------------------------------------------------------------------
# 2) 固定 M 求最优分解（无 1）
# ----------------------------------------------------------------------
def best_factorization(N: int, M: int) -> Tuple[List[int], int]:
    """
    返回 (最优因子列表, Cover)。若无合法分解则抛异常。
    评价优先级：Cover → 重复少 → 字典序。
    """
    best_seq: Optional[Tuple[int, ...]] = None
    best_key  = None           # (Cover, -dup, seq)

    def dfs(prefix: Tuple[int, ...], rem: int, lo: int, slots: int):
        nonlocal best_seq, best_key
        if slots == 1:
            if rem >= lo:
                seq  = prefix + (rem,)
                cover = reachable_products(seq)
                dup   = len(seq) - len(set(seq))
                key   = (cover, -dup, seq)
                if best_key is None or key > best_key:
                    best_seq, best_key = seq, key
            return
        for d in divisors(rem):
            if d < max(2, lo) or rem % d:
                continue
            dfs(prefix + (d,), rem // d, d, slots - 1)

    dfs(tuple(), N, 2, M)
    if best_seq is None:
        raise ValueError(f"No factorization of {N} into {M} factors ≥2.")
    return list(best_seq), best_key[0]

# Not use
# ----------------------------------------------------------------------
# 3) 自适应选择 M  (gamma 饱和式为默认)
# ----------------------------------------------------------------------
def adaptive_factorization(
    DimSize: int,
    gamma: float = 0.9,
    lam: Optional[float] = None,   # 若给出则走 λ-目标函数
) -> Tuple[int, List[int], int]:
    """
    返回 (选出的 M, 因子列表, Cover)。
    * 默认 gamma-饱和：最小 M 使 Cover ≥ gamma * Cover_max
    * 若指定 lam，则最大化 Cover - lam * 2^M
    """
    if DimSize <= 1:
        return 1, [DimSize], 1, 1
    pf = prime_factors(DimSize)
    P  = len(pf)                 # <= 60 对 64 位整数足够
    cover_max = reachable_products(tuple(pf))

    best_M, best_seq, best_cover = None, None, None
    best_obj = float("-inf")

    for M in range(2, P + 1):
        try:
            seq, cover = best_factorization(DimSize, M)
        except ValueError:
            continue

        # ---- 决策标准 ----
        if lam is None:          # γ-饱和
            if cover >= gamma * cover_max:
                return M, seq, cover, cover_max
        else:                    # λ-目标函数
            obj = cover - lam * (1 << M)
            if obj > best_obj:
                best_obj = obj
                best_M, best_seq, best_cover = M, seq, cover

    if lam is None:              # γ-饱和但未达到阈值 → 返回最大 Cover
        return P, pf, cover_max, cover_max
    else:
        if best_seq is None:
            raise RuntimeError("No feasible factorization found.")
        return best_M, best_seq, best_cover, cover_max

# Not use
def manual_factorization(DimSize: int) -> List[int]:
    if DimSize <= 128:
        return adaptive_factorization(DimSize=DimSize)[1]
    else:
        return best_factorization(DimSize, 5)[0]


# FlexibleFactorization is used by MIREDO in ASPDAC2026
def flexible_factorization(N: int, M: int) -> list[int]:
    """
    Adaptive factorisation for data-flow optimisation.

    Parameters
    ----------
    N : int
        Dimension size (positive integer; N<=1 returns [N]).
    M : int
        Memory-placement depth: the maximum number of levels across which a
        dimension's temporal factors may be distributed. Set by the hardware
        (deepest per-operand admissible chain), NOT a free constant. The HC
        invariant preserves every 1..M-level placement, so M must match the
        hierarchy or merges can drop a placement the solver could actually reach.

    Returns
    -------
    list[int]
        Ascending factor list whose product = N.
        • Tile-size set T(F) = D(N) preserved by hard constraint;
        • Multi-level tiling configurations C_k(F) = C_k(P) preserved for k=1..M;
        • Factor count minimized subject to the above two invariants;
        • Deterministic and reproducible.
    """

    if N <= 1:
        # raise ValueError("N must be an integer > 1")
        return [N]

    # ── 1. 素因子分解 ───────────────────────────────────────────
    def prime_factors(x: int) -> list[int]:
        pf, p = [], 2
        while p * p <= x:
            while x % p == 0:
                pf.append(p)
                x //= p
            p += 1 if p == 2 else 2
        if x > 1:
            pf.append(x)
        return pf                # 升序

    factors = prime_factors(N)

    # ── 2. 层次覆盖度 (HC) 评分：k = 1..M 级（M = 硬件放置深度）───────────────
    @lru_cache(maxsize=None)
    def hc_score(fs: tuple[int, ...]) -> float:
        fs = list(fs)
        n = len(fs)
        buckets = [set() for _ in range(M)]   # k = 1..M

        def dfs(idx: int, groups: list[list[int]]):
            if idx == n:
                k = len(groups)
                if 1 <= k <= M:
                    tpl = tuple(sorted(math.prod(g) for g in groups))
                    buckets[k-1].add(tpl)
                return
            # 放进已有 group
            for g in groups:
                g.append(fs[idx])
                dfs(idx + 1, groups)
                g.pop()
            # 新建 group（≤M 级）
            if len(groups) < M:
                groups.append([fs[idx]])
                dfs(idx + 1, groups)
                groups.pop()

        dfs(0, [])
        # 权重 2^-i 递减；正权递减下 HC 守恒 <=> 每个 |K_k| 守恒（合并只会塌缩表示）
        return sum(len(buckets[i]) / (2 ** i) for i in range(M))

    hc_full = hc_score(tuple(factors))   # 理论满分

    # ── Tile-size completeness check ──────────────────────────
    def _tile_set(fs: list[int]) -> set[int]:
        """All distinct products of subsets of *fs*."""
        ts = {1}
        for f in fs:
            ts |= {t * f for t in ts}
        return ts

    # ── 3. 贪婪合并：仅在 tile-set 和 HC-score 均不变时执行 ──
    MIN_FACTORS = 2

    base_tiles = _tile_set(factors)

    while len(factors) > MIN_FACTORS:
        base_hc = hc_score(tuple(sorted(factors)))
        best_new = None

        for i, j in combinations(range(len(factors)), 2):
            merged = factors[:i] + factors[i+1:j] + factors[j+1:]
            merged.append(factors[i] * factors[j])
            merged.sort()
            if _tile_set(merged) != base_tiles:          # tile-set invariant
                continue
            if hc_score(tuple(merged)) < base_hc:        # HC-score invariant
                continue
            if best_new is None or len(merged) < len(best_new):
                best_new = merged

        if best_new is None:
            break
        factors = best_new

    return factors

# ---------- 示例 ----------
if __name__ == "__main__":
    # N = 56
    # for N in [3,7,14,28,56,64,128,256,512]:
    #     M, seq, cover, cover_max = adaptive_factorization(DimSize=N)
    #     print(f"{N} 在 {M} 个因子下的最优分解：{seq}, 覆盖范围:{cover}, 最大范围:{cover_max}")
    for N in [112,1,3,7,14,28,56,64,128,256,512]:
        print(f"{N}的最优分解因子为：")
        if N > 1:
            print(f"-----prime_factorization: {prime_factors(N)}")
        print(f"-----adaptive_factorization: {adaptive_factorization(N)[1]}")
        print(f"-----manual_factorization:   {manual_factorization(N)}")
        
        # FlexibleFactorization is used by MIREDO in ASPDAC2026
        print(f"-----best_factorization:     {flexible_factorization(N, M=4)}") 