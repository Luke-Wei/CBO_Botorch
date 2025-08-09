"""
Regret calculation utilities for CBO vs BO experiments
"""

import numpy as np


def get_true_optimal_value(graph_type):
    """
    Get the theoretical optimal value for each benchmark graph.
    
    Args:
        graph_type (str): The type of graph ('ToyGraph', 'CompleteGraph', etc.)
        
    Returns:
        float: The theoretical optimal value
    """
    
    # 基于各个基准的SEM定义，这些是理论最优值
    # 注意：这些值是通过数学分析或大量采样获得的理论最优
    theoretical_optima = {
        'ToyGraph': -0.615781,       # ToyGraph的理论最优（最大化问题，取负号）
        'CompleteGraph': 0.0,        # CompleteGraph: sum of (x-1)^2，最优在x=1时为0
        'CoralGraph': 0.0,           # CoralGraph: sum of (x-2)^2 + 0.1*sum(sin(x))，近似最优
        'SimplifiedCoralGraph': 0.0  # SimplifiedCoralGraph: 同CoralGraph
    }
    
    return theoretical_optima.get(graph_type, 0.0)


def calculate_instantaneous_regret(y_value, graph_type, task='min'):
    """
    Calculate instantaneous regret for a single evaluation.
    
    Regret = |current_value - optimal_value|
    For minimization: regret = current_value - optimal_value
    For maximization: regret = optimal_value - current_value
    
    Args:
        y_value (float): Current function value
        graph_type (str): The benchmark graph type
        task (str): 'min' or 'max'
        
    Returns:
        float: Instantaneous regret (always >= 0)
    """
    optimal_value = get_true_optimal_value(graph_type)
    
    if task == 'min':
        # For minimization problems: regret = current - optimal
        regret = y_value - optimal_value
    else:
        # For maximization problems: regret = optimal - current
        regret = optimal_value - y_value
    
    # Regret should always be non-negative
    return max(0.0, regret)


def calculate_simple_regret(current_best_value, graph_type, task='min'):
    """
    Calculate simple regret (best-so-far regret).
    
    Args:
        current_best_value (float): Best value found so far
        graph_type (str): The benchmark graph type
        task (str): 'min' or 'max'
        
    Returns:
        float: Simple regret
    """
    optimal_value = get_true_optimal_value(graph_type)
    
    if task == 'min':
        # For minimization: regret = current_best - optimal
        regret = current_best_value - optimal_value
    else:
        # For maximization: regret = optimal - current_best  
        regret = optimal_value - current_best_value
    
    return max(0.0, regret)


def determine_task_type(graph_type):
    """
    Determine if the benchmark is a minimization or maximization problem.
    
    Args:
        graph_type (str): The benchmark graph type
        
    Returns:
        str: 'min' or 'max'
    """
    # 基于代码分析，ToyGraph是最大化问题（取负号），其他是最小化
    if graph_type == 'ToyGraph':
        return 'max'  # ToyGraph maximizes Y = cos(Z) - exp(-Z/20)
    else:
        return 'min'  # 其他都是最小化问题