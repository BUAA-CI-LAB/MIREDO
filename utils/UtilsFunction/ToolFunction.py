import os
import configparser
import sympy
import pathlib
import itertools
import time
import json
import subprocess
import glob
from functools import reduce

func_conv_info = {}
def _getDim(input_shape, weight_shape, stride=1, pad=0):
    if len(input_shape) != 4 or len(weight_shape) != 4:
        raise ValueError("Expected input_shape and weight_shape to have 4 dimensions")
    N, C, H, W = input_shape
    out_channels, in_channels, filter_h, filter_w = weight_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    return N * out_h * out_w, C * filter_h * filter_w, out_channels

def _getIm2colShape(input_shape, weight_shape, stride=1, pad=0):
    if len(input_shape) != 4 or len(weight_shape) != 4:
        raise ValueError("Expected input_shape and weight_shape to have 4 dimensions")
    N, C, H, W = input_shape
    out_channels, in_channels, filter_h, filter_w = weight_shape

    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    im2col_input_shape = (N * out_h * out_w, C * filter_h * filter_w)

    im2col_weight_shape = (C * filter_h * filter_w, out_channels)

    return im2col_input_shape, im2col_weight_shape

def _im2col_output_shape_groups(input_shape, weight_shape, stride=1, pad=0, groups=1):
    N, C, H, W = input_shape
    out_channels, in_channels, filter_h, filter_w = weight_shape

    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    if groups == 1:
        im2col_input_shape = (N * out_h * out_w, C * filter_h * filter_w)
        im2col_weight_shape = (out_channels, C * filter_h * filter_w)
        num_matrix_multiplications = 1
    elif groups == C:
        im2col_input_shape = (N * out_h * out_w, filter_h * filter_w)
        im2col_weight_shape = (out_channels // groups, filter_h * filter_w)
        num_matrix_multiplications = C
    else:
        im2col_input_shape = (N * out_h * out_w, (C // groups) * filter_h * filter_w)
        im2col_weight_shape = (out_channels // groups, (C // groups) * filter_h * filter_w)
        num_matrix_multiplications = groups

    return im2col_input_shape, im2col_weight_shape, num_matrix_multiplications

def combine_factors(factors, min_factor, max_factor):
    small_factors = [f for f in factors if f < min_factor]
    medium_factors = [f for f in factors if min_factor <= f <= max_factor]
    large_factors = [f for f in factors if f > max_factor]
    combined_factors = []

    small_factors.sort()
    temp_product = 1
    for f in small_factors:
        temp_product *= f
        if min_factor <= temp_product <= max_factor:
            combined_factors.append(temp_product)
            temp_product = 1
        elif temp_product > max_factor:
            while temp_product > max_factor:
                temp_product //= f
                combined_factors.append(f)
            if min_factor <= temp_product <= max_factor:
                combined_factors.append(temp_product)
                temp_product = 1
            else:
                temp_product = f
    if temp_product != 1:
        combined_factors.append(temp_product)

    combined_factors.extend(medium_factors)

    for f in large_factors:
        if sympy.isprime(f):
            combined_factors.append(f)
        else:
            sub_factors = getPrimeFactors(f)
            sub_combined = combine_factors(sub_factors, min_factor, max_factor)
            combined_factors.extend(sub_combined)

    final_factors = []
    for f in combined_factors:
        if f > max_factor:
            if sympy.isprime(f):
                final_factors.append(f)
            else:
                sub_factors = getPrimeFactors(f)
                sub_combined = combine_factors(sub_factors, min_factor, max_factor)
                final_factors.extend(sub_combined)
        else:
            final_factors.append(f)

    return final_factors

def prime_factorization_list(numbers, min_factor=2, max_factor=100):
    result = []
    for number in numbers:
        if sympy.isprime(number) and number > max_factor:
            number += 1
        factors = getPrimeFactors(number)
        combined = combine_factors(factors, min_factor, max_factor)
        result.append(combined)
    return result

def getDivisors(N: int) -> list[int]:
    if N <= 0:
        return []
    small_divs = []
    large_divs = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            small_divs.append(i)
            if i != N // i:
                large_divs.append(N // i)
        i += 1
    return small_divs + large_divs[::-1]

def getUniqueFactors(list_factors):
    unique_factors = set()

    for sublist in list_factors:
        for element in sublist:
            unique_factors.add(element)

    sorted_list = sorted(unique_factors)
    return sorted_list[1:] if sorted_list[0] == 1 else sorted_list

def prepare_save_dir(save_dir: str) -> pathlib.Path:
    p = pathlib.Path(save_dir).expanduser().resolve()
    if p.exists():
        if p.is_dir():
            pass
        else:
            raise ValueError(f"目标路径 {p} 已存在且是文件，无法作为日志目录！")
    else:
        p.mkdir(parents=True, exist_ok=False)
    return p

def get_code_version(repo_root=None):
    if repo_root is None:
        repo_root = os.path.join(os.path.dirname(__file__), '..', '..')
    try:
        commit = subprocess.check_output(
            ['git', 'log', '-1', '--format=%H', '--',
             'utils/SolverTSS.py', 'Simulator/Simulax.py'],
            cwd=repo_root, stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit[:12] if commit else 'unknown'
    except Exception:
        return 'unknown'


def save_result_json(result_dir, prefix, result_dict, case_key='workload'):
    os.makedirs(result_dir, exist_ok=True)
    code_ver = get_code_version()
    result_dict.setdefault('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))
    result_dict['code_version'] = code_ver
    case_id = result_dict.get(case_key, '')
    for f in glob.glob(os.path.join(result_dir, f"{prefix}_*.json")):
        try:
            with open(f) as fh:
                old = json.load(fh)
            if old.get(case_key) == case_id and old.get('code_version') != code_ver:
                os.remove(f)
        except Exception:
            pass
    import uuid
    result_file = os.path.join(result_dir, f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.json")
    with open(result_file, 'w') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    return result_file


def make_timestamped_dir(base, prefix=""):
    name = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}" if prefix else time.strftime('%Y%m%d_%H%M%S')
    p = os.path.join(base, name)
    os.makedirs(p, exist_ok=True)
    return p


def get_Spatial_Unrolling(dim, mappingRule, SpUnrolling):
    num_dims = len(dim)
    num_axes = len(mappingRule)

    all_dim_candidates = []

    for d_idx, d_val in enumerate(dim):
        active_axes = [u for u in range(num_axes) if mappingRule[u][d_idx] == 1]

        factors = getDivisors(d_val)

        candidates = []
        for p in itertools.product(factors, repeat=len(active_axes)):
            total_unroll = reduce(lambda x, y: x*y, p, 1)
            if d_val % total_unroll != 0:
                continue

            col = [1] * num_axes
            is_valid_candidate = True

            for i, axis_idx in enumerate(active_axes):
                val = p[i]
                if val > SpUnrolling[axis_idx]:
                    is_valid_candidate = False
                    break
                col[axis_idx] = val

            if is_valid_candidate:
                candidates.append(col)

        all_dim_candidates.append(candidates)

    for combination in itertools.product(*all_dim_candidates):

        scheme_by_axis = list(zip(*combination))

        valid_scheme = True
        for u in range(num_axes):
            axis_total_prod = reduce(lambda x, y: x*y, scheme_by_axis[u], 1)
            if axis_total_prod > SpUnrolling[u]:
                valid_scheme = False
                break

        if valid_scheme:
            yield [list(col) for col in scheme_by_axis]
