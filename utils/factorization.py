from functools import lru_cache
import math
from typing import List, Tuple, Optional
from itertools import combinations

def prime_factors(n: int) -> List[int]:
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
    return factors

@lru_cache(maxsize=None)
def divisors(x: int) -> Tuple[int, ...]:
    ds = []
    for i in range(1, math.isqrt(x) + 1):
        if x % i == 0:
            ds.append(i)
            if i != x // i:
                ds.append(x // i)
    return tuple(sorted(ds))

def reachable_products(factors: Tuple[int, ...]) -> int:
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
        for g in groups:
            g.append(factors[idx])
            dfs(idx + 1, groups)
            g.pop()
        groups.append([factors[idx]])
        dfs(idx + 1, groups)
        groups.pop()
    dfs(0, [])
    return len(prods)

def best_factorization(N: int, M: int) -> Tuple[List[int], int]:
    best_seq: Optional[Tuple[int, ...]] = None
    best_key  = None

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

def adaptive_factorization(
    DimSize: int,
    gamma: float = 0.9,
    lam: Optional[float] = None,
) -> Tuple[int, List[int], int]:
    if DimSize <= 1:
        return 1, [DimSize], 1, 1
    pf = prime_factors(DimSize)
    P  = len(pf)
    cover_max = reachable_products(tuple(pf))

    best_M, best_seq, best_cover = None, None, None
    best_obj = float("-inf")

    for M in range(2, P + 1):
        try:
            seq, cover = best_factorization(DimSize, M)
        except ValueError:
            continue

        if lam is None:
            if cover >= gamma * cover_max:
                return M, seq, cover, cover_max
        else:
            obj = cover - lam * (1 << M)
            if obj > best_obj:
                best_obj = obj
                best_M, best_seq, best_cover = M, seq, cover

    if lam is None:
        return P, pf, cover_max, cover_max
    else:
        if best_seq is None:
            raise RuntimeError("No feasible factorization found.")
        return best_M, best_seq, best_cover, cover_max

def manual_factorization(DimSize: int) -> List[int]:
    if DimSize <= 128:
        return adaptive_factorization(DimSize=DimSize)[1]
    else:
        return best_factorization(DimSize, 5)[0]


def flexible_factorization(N: int, M: int) -> list[int]:

    if N <= 1:
        return [N]

    def prime_factors(x: int) -> list[int]:
        pf, p = [], 2
        while p * p <= x:
            while x % p == 0:
                pf.append(p)
                x //= p
            p += 1 if p == 2 else 2
        if x > 1:
            pf.append(x)
        return pf

    factors = prime_factors(N)

    @lru_cache(maxsize=None)
    def hc_score(fs: tuple[int, ...]) -> float:
        fs = list(fs)
        n = len(fs)
        buckets = [set() for _ in range(M)]

        def dfs(idx: int, groups: list[list[int]]):
            if idx == n:
                k = len(groups)
                if 1 <= k <= M:
                    tpl = tuple(sorted(math.prod(g) for g in groups))
                    buckets[k-1].add(tpl)
                return
            for g in groups:
                g.append(fs[idx])
                dfs(idx + 1, groups)
                g.pop()
            if len(groups) < M:
                groups.append([fs[idx]])
                dfs(idx + 1, groups)
                groups.pop()

        dfs(0, [])
        return sum(len(buckets[i]) / (2 ** i) for i in range(M))

    hc_full = hc_score(tuple(factors))

    def _tile_set(fs: list[int]) -> set[int]:
        ts = {1}
        for f in fs:
            ts |= {t * f for t in ts}
        return ts

    MIN_FACTORS = 2

    base_tiles = _tile_set(factors)

    while len(factors) > MIN_FACTORS:
        base_hc = hc_score(tuple(sorted(factors)))
        best_new = None

        for i, j in combinations(range(len(factors)), 2):
            merged = factors[:i] + factors[i+1:j] + factors[j+1:]
            merged.append(factors[i] * factors[j])
            merged.sort()
            if _tile_set(merged) != base_tiles:
                continue
            if hc_score(tuple(merged)) < base_hc:
                continue
            if best_new is None or len(merged) < len(best_new):
                best_new = merged

        if best_new is None:
            break
        factors = best_new

    return factors

if __name__ == "__main__":
    for N in [112,1,3,7,14,28,56,64,128,256,512]:
        print(f"{N}的最优分解因子为：")
        if N > 1:
            print(f"-----prime_factorization: {prime_factors(N)}")
        print(f"-----adaptive_factorization: {adaptive_factorization(N)[1]}")
        print(f"-----manual_factorization:   {manual_factorization(N)}")

        print(f"-----best_factorization:     {flexible_factorization(N, M=4)}")
