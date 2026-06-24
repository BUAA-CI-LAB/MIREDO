# this file is prepared for project 026
# Created by iboxl
# Modified in project 112

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
    # input_shape: (N, C, H, W)                                                     #  N = batch
    # weight_shape: (out_channels, in_channels, filter_h, filter_w)
    if len(input_shape) != 4 or len(weight_shape) != 4:
        raise ValueError("Expected input_shape and weight_shape to have 4 dimensions")
    N, C, H, W = input_shape
    out_channels, in_channels, filter_h, filter_w = weight_shape
    # 计算输出特征图的形状
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    return N * out_h * out_w, C * filter_h * filter_w, out_channels

def _getIm2colShape(input_shape, weight_shape, stride=1, pad=0):
    # input_shape: (N, C, H, W)
    # weight_shape: (out_channels, in_channels, filter_h, filter_w)
    if len(input_shape) != 4 or len(weight_shape) != 4:
        raise ValueError("Expected input_shape and weight_shape to have 4 dimensions")
    N, C, H, W = input_shape
    out_channels, in_channels, filter_h, filter_w = weight_shape
    
    # 计算输出特征图的形状
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    
    # 计算经过 im2col 操作后的输入特征图的形状
    im2col_input_shape = (N * out_h * out_w, C * filter_h * filter_w)
    
    # 计算经过 im2col 操作后的权重的形状
    im2col_weight_shape = (C * filter_h * filter_w, out_channels)
    
    return im2col_input_shape, im2col_weight_shape

def _im2col_output_shape_groups(input_shape, weight_shape, stride=1, pad=0, groups=1):
    N, C, H, W = input_shape
    out_channels, in_channels, filter_h, filter_w = weight_shape
    
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    
    # 计算 im2col 形状和矩阵乘法的个数
    if groups == 1:
        # 普通卷积
        im2col_input_shape = (N * out_h * out_w, C * filter_h * filter_w)
        im2col_weight_shape = (out_channels, C * filter_h * filter_w)
        num_matrix_multiplications = 1
    elif groups == C:
        # 深度卷积
        im2col_input_shape = (N * out_h * out_w, filter_h * filter_w)
        im2col_weight_shape = (out_channels // groups, filter_h * filter_w)
        num_matrix_multiplications = C
    else:
        # 分组卷积
        im2col_input_shape = (N * out_h * out_w, (C // groups) * filter_h * filter_w)
        im2col_weight_shape = (out_channels // groups, (C // groups) * filter_h * filter_w)
        num_matrix_multiplications = groups
    
    return im2col_input_shape, im2col_weight_shape, num_matrix_multiplications

def combine_factors(factors, min_factor, max_factor):
    small_factors = [f for f in factors if f < min_factor]
    medium_factors = [f for f in factors if min_factor <= f <= max_factor]
    large_factors = [f for f in factors if f > max_factor]
    combined_factors = []

    # 合并小于 min_factor 的因子
    small_factors.sort()
    temp_product = 1
    for f in small_factors:
        temp_product *= f
        if min_factor <= temp_product <= max_factor:
            combined_factors.append(temp_product)
            temp_product = 1
        elif temp_product > max_factor:
            # 无法再合并，分解 temp_product
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

    # 添加中间的因子
    combined_factors.extend(medium_factors)

    # 处理大于 max_factor 的因子
    for f in large_factors:
        # 尝试将大因子分解为较小的因子
        if sympy.isprime(f):
            # 如果是素数，无法分解，直接添加
            combined_factors.append(f)
        else:
            # 分解为因子
            sub_factors = getPrimeFactors(f)
            # 递归合并这些因子
            sub_combined = combine_factors(sub_factors, min_factor, max_factor)
            combined_factors.extend(sub_combined)

    # 最终结果中可能有因子不在范围内，需要再次检查
    final_factors = []
    for f in combined_factors:
        if f > max_factor:
            # 尝试进一步分解
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
        # 检查是否为素数，且大于 8
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
    # 只需遍历到 sqrt(N)
    while i * i <= N:
        if N % i == 0:
            small_divs.append(i)
            if i != N // i:
                large_divs.append(N // i)
        i += 1
    return small_divs + large_divs[::-1]

def getUniqueFactors(list_factors):
    """
    从嵌套列表 nested_list 中提取所有整数，去重并升序排序后返回。

    参数:
        nested_list (list of list of int): 形如 [[3], [2,4], [2,7,8], [4,7]] 的输入
    返回:
        list of int: 升序排列的唯一元素列表，例如 [2,3,4,7,8]
    """
    # 使用一个 set 结构用于去重
    unique_factors = set()

    # 遍历每个子列表，将其元素加入集合
    for sublist in list_factors:
        # 这里假设 sublist 中的所有元素均为 int 类型
        for element in sublist:
            unique_factors.add(element)

    # 将去重后的集合转换为列表并排序
    sorted_list = sorted(unique_factors)
    # return sorted_list
    return sorted_list[1:] if sorted_list[0] == 1 else sorted_list

def prepare_save_dir(save_dir: str) -> pathlib.Path:
    """
    检查 save_dir 路径是否可用：
        1. 若路径已经存在且为文件夹，原样使用；
        2. 若路径不存在，则递归创建文件夹；
        3. 若路径存在但为普通文件，抛出 ValueError。
    返回值
    ------
    pathlib.Path
        规范化后的绝对路径对象
    """
    p = pathlib.Path(save_dir).expanduser().resolve()  # 支持 ~、相对路径
    if p.exists():
        if p.is_dir():
            pass  # 目录已存在，可直接使用
        else:
            raise ValueError(f"目标路径 {p} 已存在且是文件，无法作为日志目录！")
    else:
        p.mkdir(parents=True, exist_ok=False)  # 递归创建
    return p

def get_code_version(repo_root=None):
    """获取影响实验结果的关键源文件的git版本标识（commit hash前12位）。
    跟踪 SolverTSS.py 和 Simulax.py — 这两个文件的变更会影响MIP和simulator的行为。"""
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
    """保存实验结果JSON，自动附加code_version和timestamp。
    仅清理同case、不同code_version的旧结果；同版本多次运行结果全部保留。
    参数:
        result_dir: 保存目录 (如 output/Eval_Result/)
        prefix: 文件名前缀 (如 'bruteforce', 'enumLoop')
        result_dict: 结果字典（会自动补充timestamp和code_version）
        case_key: result_dict中标识case的字段名
    返回: 保存的文件路径"""
    os.makedirs(result_dir, exist_ok=True)
    code_ver = get_code_version()
    result_dict.setdefault('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))
    result_dict['code_version'] = code_ver
    # 清理同case异版本旧结果
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
    """创建带时间戳的输出目录，格式: base/prefix_YYYYMMDD_HHMMSS。
    用于 run.py / SolveMapping.py 等需要唯一输出目录的场景。"""
    name = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}" if prefix else time.strftime('%Y%m%d_%H%M%S')
    p = os.path.join(base, name)
    os.makedirs(p, exist_ok=True)
    return p


def get_Spatial_Unrolling(dim, mappingRule, SpUnrolling):
    """
    dim: 维度大小列表
    mappingRule: [轴][维度] 的映射矩阵
    SpUnrolling: 各轴最大展开限制

    # generator = solve_unrolling(dim, mappingRule, SpUnrolling)
    # for scheme in generator:
    #     ...
    """
    num_dims = len(dim)
    num_axes = len(mappingRule)
    
    # --- 第1步：生成每个维度各自的合法“列”方案 ---
    # 结果结构：all_dim_candidates[d] = [[u0, u1, u2], [u0', u1', u2']...]
    all_dim_candidates = []
    
    for d_idx, d_val in enumerate(dim):
        # 1.1 找出该维度允许映射到的轴
        active_axes = [u for u in range(num_axes) if mappingRule[u][d_idx] == 1]
        
        # 1.2 获取该维度的所有因子 (作为可能的展开大小)
        factors = getDivisors(d_val)
        
        candidates = []
        # 生成因子的笛卡尔积。例如该维度映射到2个轴，就找 (f1, f2)
        # 只有当 mappingRule 允许该维度映射到某轴时，才从因子中取值，否则强制为1
        for p in itertools.product(factors, repeat=len(active_axes)):
            # 检查A：展开因子的乘积必须能整除维度大小
            total_unroll = reduce(lambda x, y: x*y, p, 1)
            if d_val % total_unroll != 0:
                continue
                
            # 构造完整的列向量 (对应每个轴的展开大小)
            col = [1] * num_axes
            is_valid_candidate = True
            
            for i, axis_idx in enumerate(active_axes):
                val = p[i]
                # 检查B (单维度预剪枝)：如果单维度的因子已经超过了该轴的总限制，直接丢弃
                if val > SpUnrolling[axis_idx]:
                    is_valid_candidate = False
                    break
                col[axis_idx] = val
            
            if is_valid_candidate:
                candidates.append(col)
        
        all_dim_candidates.append(candidates)

    # --- 第2步：组合所有维度 (笛卡尔积) ---
    # itertools.product 会从每个 all_dim_candidates[i] 中选一个 col 拼成完整方案
    for combination in itertools.product(*all_dim_candidates):
        # combination 结构是 tuple(col_dim0, col_dim1, ...)
        # 我们需要按轴汇总，检查总乘积是否超标
        
        # 转置矩阵：将 (维度, 轴) 变为 (轴, 维度)
        # zip(*combination) 实现了矩阵转置
        scheme_by_axis = list(zip(*combination)) 
        
        # --- 第3步：全局合法性检查 ---
        valid_scheme = True
        for u in range(num_axes):
            # 计算该轴上所有维度展开因子的乘积
            axis_total_prod = reduce(lambda x, y: x*y, scheme_by_axis[u], 1)
            if axis_total_prod > SpUnrolling[u]:
                valid_scheme = False
                break
        
        if valid_scheme:
            # 转换回 list of lists 格式以便输出
            yield [list(col) for col in scheme_by_axis]
