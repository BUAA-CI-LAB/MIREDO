# this file is prepared for project 026
# Created by iboxl

import math
import gurobipy as gp
from gurobipy import GRB, quicksum
from utils.GlobalUT import *

def var_eq(model:gp.Model, expr, vtype):
    var = model.addVar(vtype=vtype)
    model.addConstr(var == expr)
    return var

def var_le(model:gp.Model, expr, vtype):
    var = model.addVar(vtype=vtype)
    model.addConstr(var <= expr)
    return var

def var_ge(model:gp.Model, expr, vtype):
    var = model.addVar(vtype=vtype)
    model.addConstr(var >= expr)
    return var

def var_AmulBeqC(model:gp.Model, vtype, A, B):
    var = model.addVar(vtype=vtype)
    if isinstance(A, gp.QuadExpr) or isinstance(A, gp.LinExpr):
        var_A = model.addVar(vtype=GRB.CONTINUOUS)
        model.addConstr(var_A==A)
    else:
        var_A = A
    model.addConstr(var * var_A == B)
    return var

def var_AmulBleC(model:gp.Model, vtype, B, C):
    var = model.addVar(vtype=vtype)
    model.addConstr(var * B <= C)
    return var

def var_AmulBgeC(model:gp.Model, vtype, B, C):
    var = model.addVar(vtype=vtype)
    model.addConstr(var * B >= C)
    return var

def var_max(model:gp.Model, vtype, A, B):
    if isinstance(A, gp.QuadExpr) or isinstance(A, gp.LinExpr):
        var_A = model.addVar(vtype=vtype)
        model.addConstr(var_A==A)
    else:
        var_A = A
    var = model.addVar(vtype=vtype)
    model.addConstr(var >= var_A)
    model.addConstr(var >= B)
    # model.addGenConstrMax(var, [var_A, B], 0)
    return var

def var_max_const(model:gp.Model, vtype, B, const):
    if not (isinstance(B, gp.Var) or isinstance(B, gp.QuadExpr) or isinstance(B, gp.LinExpr)):
        return max(B, const)
    if isinstance(B, gp.QuadExpr) or isinstance(B, gp.LinExpr):
        var_B = model.addVar(vtype=GRB.CONTINUOUS)
        model.addConstr(var_B==B)
    else:
        var_B = B
    var = model.addVar(vtype=vtype)
    model.addGenConstrMax(var, [var_B], const)
    return var

def var_min(model:gp.Model, vtype, A, B):
    if isinstance(A, gp.QuadExpr) or isinstance(A, gp.LinExpr):
        var_A = model.addVar(vtype=GRB.CONTINUOUS)
        model.addConstr(var_A==A)
    else:
        var_A = A
    if isinstance(B, gp.QuadExpr) or isinstance(B, gp.LinExpr):
        var_B = model.addVar(vtype=GRB.CONTINUOUS)
        model.addConstr(var_B==B)
    else:
        var_B = B
    var = model.addVar(vtype=vtype)
    model.addGenConstrMin(var, [var_A, var_B], 5000)
    return var

def var_min_const(model:gp.Model, vtype, B, const):
    if not isinstance(B, gp.Var):
        return min(B, const)
    var = model.addVar(vtype=vtype)
    model.addGenConstrMin(var, [B], const)
    return var

def var_add(model:gp.Model, vtype, A, B):
    var = model.addVar(vtype=vtype)
    model.addConstr(var==A+B)
    return var

def exp_add(model:gp.Model, vtype, A, B):
    return A+B

def var_minus(model:gp.Model, vtype, B, C):
    return B-C
    # var = model.addVar(vtype=vtype)             # check B >= C
    # model.addConstr(var==B-C)
    # return var

def var_mul(model:gp.Model, vtype, A, B):
    if isinstance(A, gp.QuadExpr) or isinstance(A, gp.LinExpr):
        var_A = model.addVar(lb=0, vtype=vtype)
        model.addConstr(var_A==A)
    else:
        var_A = A
    if isinstance(B, gp.QuadExpr) or isinstance(B, gp.LinExpr):
        var_B = model.addVar(lb=0, vtype=vtype)
        model.addConstr(var_B==B)
    else:
        var_B = B
    if isinstance(var_A, gp.Var) and isinstance(var_B, gp.Var):
        model.update()
        newub = var_A.ub * var_B.ub
        var = model.addVar(lb=0, ub=newub, vtype=vtype)
        model.addConstr(var==var_A * var_B)
    else:
        var = var_A * var_B
    return var

def var_mulABC(model:gp.Model, vtype, A, B, C):
    tmp = var_mul(model=model, vtype=vtype, A=A, B=B)
    var = var_mul(model=model, vtype=vtype, A=tmp, B=C)
    return var

def exp_mul(model:gp.Model, vtype, A, B):
    return A * B

def var_AgeB(model:gp.Model, A, B):
    var = model.addVar(vtype=GRB.BINARY)             # check B >= C
    if isinstance(A, gp.QuadExpr) or isinstance(A, gp.LinExpr):
        var_A = model.addVar(vtype=GRB.CONTINUOUS)
        model.addConstr(var_A==A)
    else:
        var_A = A
    if isinstance(B, gp.QuadExpr) or isinstance(B, gp.LinExpr):
        var_B = model.addVar(vtype=GRB.CONTINUOUS)
        model.addConstr(var_B==B)
    else:
        var_B = B
    if (not isinstance(var_A, gp.Var)) and (not isinstance(var_B, gp.Var)):
        if var_A >= var_B:
            model.addConstr(var==True)
        else:
            model.addConstr(var==False)
        return var, var_A, var_B
    model.addGenConstrIndicator(var, True, var_A>=var_B)
    model.addGenConstrIndicator(var, False, var_A<=var_B-1)
    return var, var_A, var_B

def var_AleB(model:gp.Model, A, B):
    var = model.addVar(vtype=GRB.BINARY)             # check B >= C
    if isinstance(A, gp.QuadExpr) or isinstance(A, gp.LinExpr):
        var_A = model.addVar(vtype=GRB.CONTINUOUS)
        model.addConstr(var_A==A)
    else:
        var_A = A
    if isinstance(B, gp.QuadExpr) or isinstance(B, gp.LinExpr):
        var_B = model.addVar(vtype=GRB.CONTINUOUS)
        model.addConstr(var_B==B)
    else:
        var_B = B
    if (not isinstance(var_A, gp.Var)) and (not isinstance(var_B, gp.Var)):
        if var_A <= var_B:
            model.addConstr(var==True)
        else:
            model.addConstr(var==False)
        return var, var_A, var_B
    model.addGenConstrIndicator(var, True, var_A<=var_B)
    model.addGenConstrIndicator(var, False, var_A>=var_B+1)
    return var, var_A, var_B

def var_AandB(model, A, B, name):
    if not isinstance(A, gp.Var) and not isinstance(B, gp.Var):
        return 1.0 if A and B else 0.0
    elif not isinstance(A, gp.Var):
        return B if A==True else 0.0
    elif not isinstance(B, gp.Var):
        return A if B==True else 0.0
    else:
        var = model.addVar(vtype=GRB.BINARY, name=f'indic_{name}') 
        model.addConstr( var <= A, name=f'C_AandB_{name}_1')
        model.addConstr( var <= B, name=f'C_AandB_{name}_2')
        model.addConstr( var >= A + B - 1, name=f'C_AandB_{name}_3')
        return var

def var_AorB(model, A, B, name):
    var = model.addVar(vtype=GRB.BINARY, name=f'indic_{name}')
    model.addConstr(var >= A, name=f'C_AorB_{name}_1')
    model.addConstr(var >= B, name=f'C_AorB_{name}_2')
    model.addConstr(var <= A + B, name=f'C_AorB_{name}_3')
    return var

def tag_constraint(model:gp.Model, tag, num, name):
    num_block = model.addVar(lb=1, ub=num, vtype=GRB.INTEGER, name=name)
    model.addConstrs(tag[i]>=tag[i+1] for i in range(num-1))                # 可以注释
    model.addConstr(quicksum(tag[i] for i in range(num)) == num_block)
    return num_block

def get_startVar_byConstr(model:gp.Model, constraint_hypothesis):
    model.optimize()
    for constr in constraint_hypothesis:
        model.remove(constr)
    if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
        optimal_values = {}
        for var in model.getVars():
            optimal_values[var.VarName] = var.X
        model.update()
        return optimal_values
    return None

def get_startVar_set(model:gp.Model):
    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
        optimal_values = {}
        for var in model.getVars():
            optimal_values[var.VarName] = var.X
        model.update()
        return optimal_values
    return None

def set_startVar(model:gp.Model, preset_conditions, res_str):
    best_sv = None
    best_l = float('inf')  # 初始化为无穷大
    for sv in preset_conditions:
        if sv is None:  # 跳过 None 值
            continue
        l = sv.get(res_str, None)
        if l is not None and l < best_l:
            best_l = l
            best_sv = sv
    if best_sv is not None:
        for var in model.getVars():
            var.Start = best_sv.get(var.VarName, None)
        Logger.debug(f"set_startVar with {res_str}= {best_l}")
    else:
        Logger.debug("No startVar Found")

def var_mul01(model:gp.Model, indic, A, name, A_ub = None, A_lb = None, var_ub = None):
    if not isinstance(indic, gp.Var):
        if indic==True:
            return A
        else:
            return 0
    # model.update() flushes all pending Gurobi operations — expensive when
    # called repeatedly.  Only needed when we read back Var.ub or Var.lb
    # from the model (bounds may have been modified after addVar).
    need_ub = A_ub is None and isinstance(A, gp.Var)
    need_lb = A_lb is None and isinstance(A, gp.Var)
    if need_ub or need_lb:
        model.update()
    ub = A_ub if A_ub is not None else (A.ub if isinstance(A, gp.Var) else A)
    lb = A_lb if A_lb is not None else (A.lb if isinstance(A, gp.Var) else 0)

    var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=(ub if var_ub is None else var_ub), name=name)
    model.addConstr(var <= ub * indic, name=f"C_{name}_1")
    model.addConstr(var >= A - ub * (1 - indic), name=f"C_{name}_2")
    model.addConstr(var >= lb * indic, name=f"C_{name}_3")
    model.addConstr(var <= A - lb * (1 - indic), name=f"C_{name}_4")
    return var

def var_exp(model:gp.Model, lg_term, lb, ub, name):
    var = model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=name)
    model.addGenConstrExp(xvar=lg_term, yvar=var, options=CONST.ExpOption, name=f"C_{name}")
    return var