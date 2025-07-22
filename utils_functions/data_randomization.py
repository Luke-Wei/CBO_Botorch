#!/usr/bin/env python3
"""
Data randomization functions - ensure BoTorch and GPy versions use the same randomization strategy
"""

import numpy as np
import torch
import sys
import os
from typing import Tuple, Union

# Add project path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def randomize_initial_data(
    data_x: Union[np.ndarray, torch.Tensor], 
    data_y: Union[np.ndarray, torch.Tensor], 
    num_interventions: int, 
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Randomize initial data using exactly the same method as GPy version
    
    This function replicates the logic from utils_functions/BO_functions.py define_initial_data_BO
    
    Args:
        data_x: Input data (N, input_dim)
        data_y: Output data (N, 1)
        num_interventions: Number of required interventions
        seed: Random seed
        
    Returns:
        Tuple of (randomized_data_x, randomized_data_y, min_intervention_value, min_y)
    """
    
    # Convert to numpy arrays for compatibility
    if torch.is_tensor(data_x):
        data_x = data_x.numpy()
    if torch.is_tensor(data_y):
        data_y = data_y.numpy()
    
    # Ensure correct shape
    if len(data_x.shape) == 1:
        data_x = data_x[:, np.newaxis]
    if len(data_y.shape) == 1:
        data_y = data_y[:, np.newaxis]
    
    # Combine data - exactly same as GPy version
    all_data = np.concatenate((data_x, data_y), axis=1)
    
    # Save current random state - exactly same as GPy version
    state = np.random.get_state()
    
    # Use seed to shuffle randomly - exactly same as GPy version
    np.random.seed(seed)
    np.random.shuffle(all_data)
    
    # Restore random state - exactly same as GPy version
    np.random.set_state(state)
    
    # Extract randomized data
    input_dim = data_x.shape[1]
    randomized_data_x = all_data[:num_interventions, :input_dim]
    randomized_data_y = all_data[:num_interventions, input_dim:]
    
    # Find minimum value and corresponding intervention value - exactly same as GPy version
    min_y = np.min(randomized_data_y)
    min_idx = np.where(randomized_data_y == min_y)[0][0]
    min_intervention_value = all_data[min_idx, :input_dim]
    
    if len(min_intervention_value.shape) == 1:
        min_intervention_value = min_intervention_value[:, np.newaxis].T
    
    return randomized_data_x, randomized_data_y, min_intervention_value, min_y


def randomize_botorch_data_lists(
    interventional_data: np.ndarray,
    exploration_set: list,
    num_interventions: int,
    seed: int
) -> Tuple[list, list, str, float, np.ndarray]:
    """
    Create randomized data lists for BoTorch version
    
    Args:
        interventional_data: Original intervention data
        exploration_set: Exploration set, like [['X'], ['Z']]
        num_interventions: Number of data points per intervention set
        seed: Random seed
        
    Returns:
        Tuple of (data_x_list, data_y_list, best_variable, opt_y, best_intervention_value)
    """
    
    data_x_list = []
    data_y_list = []
    opt_y_candidates = []
    best_intervention_values = []
    variable_names = []
    
    for i, intervention_set in enumerate(exploration_set):
        var_name = intervention_set[0]
        variable_names.append(var_name)
        
        # Find corresponding intervention data
        found_data = None
        for j, intervention in enumerate(interventional_data):
            if intervention[1] == var_name:
                found_data = intervention
                break
        
        if found_data is None:
            raise ValueError(f"Intervention data not found for variable {var_name}")
        
        # Get original data
        original_data_x = found_data[2]
        original_data_y = found_data[3]
        
        # Use seed to randomize data
        rand_data_x, rand_data_y, min_int_val, min_y = randomize_initial_data(
            original_data_x, original_data_y, num_interventions, seed
        )
        
        # Convert to torch tensor
        data_x_tensor = torch.tensor(rand_data_x, dtype=torch.float32)
        data_y_tensor = torch.tensor(rand_data_y, dtype=torch.float32)
        
        data_x_list.append(data_x_tensor)
        data_y_list.append(data_y_tensor)
        
        opt_y_candidates.append(min_y)
        best_intervention_values.append(min_int_val)
    
    # Find global optimum
    global_opt_idx = np.argmin(opt_y_candidates)
    opt_y = opt_y_candidates[global_opt_idx]
    best_variable = variable_names[global_opt_idx]
    best_intervention_value = best_intervention_values[global_opt_idx]
    
    return data_x_list, data_y_list, best_variable, opt_y, best_intervention_value


def test_randomization_consistency():
    """
    Test randomization consistency - ensure same results as GPy version
    """
    print("=== Testing Randomization Consistency ===")
    
    # Create test data
    np.random.seed(999)  # For generating test data
    test_data_x = np.random.randn(20, 1)
    test_data_y = np.random.randn(20, 1)
    
    seeds = [42, 123, 456]
    num_interventions = 10
    
    print("Testing if different seeds produce different results:")
    results = []
    
    for seed in seeds:
        rand_x, rand_y, min_val, min_y = randomize_initial_data(
            test_data_x, test_data_y, num_interventions, seed
        )
        results.append((min_y, min_val[0, 0]))
        print(f"Seed {seed}: min_y={min_y:.6f}, min_intervention={min_val[0, 0]:.6f}")
    
    # Check for differences
    unique_results = len(set([f"{r[0]:.6f}_{r[1]:.6f}" for r in results]))
    print(f"Unique results count: {unique_results}/{len(seeds)}")
    
    if unique_results == len(seeds):
        print("✅ Randomization works properly - different seeds produce different results")
    elif unique_results == 1:
        print("❌ Randomization failed - all seeds produce same results")
    else:
        print("⚠️ Partial randomization - some seeds produce same results")
    
    return unique_results == len(seeds)


if __name__ == "__main__":
    # Run test
    test_randomization_consistency()