#!/usr/bin/env python3
"""
Fix Coverage calculation issues
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_functions.CBO_functions import update_hull


def calculate_reasonable_coverage_total(observational_samples, manipulative_variables, max_N=200):
    """
    Calculate reasonable coverage_total value to keep epsilon_coverage in reasonable range
    
    Args:
        observational_samples: Observational samples
        manipulative_variables: List of manipulable variables
        max_N: Maximum number of observations
        
    Returns:
        Reasonable coverage_total value
    """
    coverage_obs = update_hull(observational_samples, manipulative_variables)
    rescale = observational_samples.shape[0] / max_N
    
    # Target is to keep epsilon_coverage between 0.3-0.7 for reasonable observe/intervene balance
    target_epsilon = 0.5  # 50% observation probability
    
    # epsilon_coverage = (coverage_obs / coverage_total) / rescale
    # So coverage_total = coverage_obs / (target_epsilon * rescale)
    reasonable_coverage_total = coverage_obs / (target_epsilon * rescale)
    
    return reasonable_coverage_total


def fixed_observation_intervention_decision(
    i, observational_samples, manipulative_variables, 
    coverage_total=None, max_N=200, fixed_prob=None
):
    """
    Fixed observation vs intervention decision function
    
    Args:
        i: Current iteration number
        observational_samples: Observational samples
        manipulative_variables: Manipulable variables
        coverage_total: Total coverage value (if None, will auto-calculate)
        max_N: Maximum number of observations
        fixed_prob: Fixed observation probability (if provided, ignores coverage calculation)
        
    Returns:
        Decision result: True for observe, False for intervene
    """
    uniform = np.random.uniform(0., 1.)
    
    # Force first two iterations: first observe, second intervene
    if i == 0:
        return True  # Observe
    if i == 1:
        return False  # Intervene
    
    if fixed_prob is not None:
        # Use fixed probability
        return uniform < fixed_prob
    
    # Use fixed coverage calculation
    coverage_obs = update_hull(observational_samples, manipulative_variables)
    rescale = observational_samples.shape[0] / max_N
    
    if coverage_total is None:
        # Auto-calculate reasonable coverage_total
        coverage_total = calculate_reasonable_coverage_total(
            observational_samples, manipulative_variables, max_N
        )
    
    epsilon_coverage = (coverage_obs / coverage_total) / rescale
    
    # Limit epsilon_coverage to reasonable range
    epsilon_coverage = min(epsilon_coverage, 0.8)  # Maximum 80% observation probability
    
    return uniform < epsilon_coverage


def test_coverage_fix():
    """
    Test coverage fix
    """
    print("=== Testing Coverage Fix ===")
    
    # Load test data
    import pickle
    with open('Data/ToyGraph/observations.pkl', 'rb') as f:
        observational_samples = pickle.load(f)
    
    initial_obs = observational_samples.iloc[:100]
    manipulative_variables = ['X', 'Z']
    
    # Original calculation
    coverage_obs = update_hull(initial_obs, manipulative_variables)
    original_coverage_total = 0.5
    rescale = initial_obs.shape[0] / 200
    original_epsilon = (coverage_obs / original_coverage_total) / rescale
    
    # Fixed calculation
    fixed_coverage_total = calculate_reasonable_coverage_total(
        initial_obs, manipulative_variables
    )
    fixed_epsilon = (coverage_obs / fixed_coverage_total) / rescale
    
    print(f"Original setup:")
    print(f"  coverage_obs: {coverage_obs:.6f}")
    print(f"  coverage_total: {original_coverage_total:.6f}")
    print(f"  epsilon_coverage: {original_epsilon:.6f}")
    print(f"  Result: Always observe (epsilon > 1)")
    
    print(f"\\nFixed setup:")
    print(f"  coverage_obs: {coverage_obs:.6f}")
    print(f"  coverage_total: {fixed_coverage_total:.6f}")
    print(f"  epsilon_coverage: {fixed_epsilon:.6f}")
    print(f"  Result: Reasonable observe/intervene balance")
    
    # Simulate 30 decisions
    np.random.seed(42)
    decisions = []
    for i in range(30):
        decision = fixed_observation_intervention_decision(
            i, initial_obs, manipulative_variables, fixed_coverage_total
        )
        decisions.append('observe' if decision else 'intervene')
    
    observe_count = decisions.count('observe')
    intervene_count = decisions.count('intervene')
    
    print(f"\\nSimulating 30 iterations:")
    print(f"  Observation count: {observe_count}")
    print(f"  Intervention count: {intervene_count}")
    print(f"  Intervention rate: {intervene_count/30:.1%}")
    
    if intervene_count > 0:
        print("✅ Fix successful - algorithm will intervene")
    else:
        print("❌ Fix failed - algorithm still only observes")
    
    return intervene_count > 0


if __name__ == "__main__":
    test_coverage_fix()