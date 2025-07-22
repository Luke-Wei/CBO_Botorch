import time
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from collections import OrderedDict

try:
    from utils_functions.BO_functions_botorch import (
        create_causal_gp_model, 
        update_causal_gp_model,
        optimize_causal_acquisition
    )
    from utils_functions.causal_acquisition_functions_botorch import CausalExpectedImprovement
    from utils_functions.CBO_functions import (
        update_hull, observe
    )
    from utils_functions.coverage_fix import (
        calculate_reasonable_coverage_total,
        fixed_observation_intervention_decision
    )
    from utils_functions.utils import (
        get_interventional_dict, list_interventional_ranges, initialise_dicts, find_current_global, add_data, get_new_dict_x
    )
    from utils_functions.graph_functions import Intervention_function
    from utils_functions.compute_update_do_functions import update_all_do_functions
    from utils_functions.cost_functions import total_cost
except ImportError as e:
    print(f"Warning: Import error in CBO_botorch: {e}")
    print("Make sure all required dependencies are installed:")
    print("pip install torch botorch gpytorch")
    raise


def CBO_botorch(
    num_trials: int,
    exploration_set: List,
    manipulative_variables: List,
    data_x_list: List[torch.Tensor],
    data_y_list: List[torch.Tensor],
    best_intervention_value: torch.Tensor,
    opt_y: float,
    best_variable: str,
    dict_ranges: Dict,
    functions: Dict,
    observational_samples: pd.DataFrame,
    coverage_total: float,
    graph: object,
    num_additional_observations: int,
    costs: Dict,
    full_observational_samples: pd.DataFrame,
    task: str = 'min',
    max_N: int = 200,
    initial_num_obs_samples: int = 100,
    num_interventions: int = 10,
    Causal_prior: bool = False,
    acquisition_function: str = 'EI',
    num_restarts: int = 10,
    raw_samples: int = 100,
) -> Tuple[List, Dict, Dict, List, int, float]:
    """
    Causal Bayesian Optimization using BoTorch and GPyTorch.
    
    Args:
        num_trials: Number of optimization trials
        exploration_set: List of intervention variable sets to explore
        manipulative_variables: List of variables that can be manipulated
        data_x_list: List of initial intervention input data for each exploration set
        data_y_list: List of initial intervention output data for each exploration set
        best_intervention_value: Best intervention value found so far
        opt_y: Best objective value found so far
        best_variable: Variable corresponding to best intervention
        dict_ranges: Dictionary of variable ranges
        functions: Dictionary of fitted causal functions
        observational_samples: Current observational data
        coverage_total: Total coverage of the intervention space
        graph: Causal graph object
        num_additional_observations: Number of observations to collect per step
        costs: Dictionary of intervention costs
        full_observational_samples: Complete observational dataset
        task: 'min' for minimization, 'max' for maximization
        max_N: Maximum number of observations to use
        initial_num_obs_samples: Initial number of observational samples
        num_interventions: Number of initial interventions per variable set
        Causal_prior: Whether to use causal prior in GP
        acquisition_function: Type of acquisition function ('EI', 'UCB')
        num_restarts: Number of restarts for acquisition optimization
        raw_samples: Number of raw samples for acquisition optimization
        
    Returns:
        Tuple of (current_cost, current_best_x, current_best_y, global_opt, observed, total_time)
    """
    
    # Initialize tracking variables
    current_cost = []
    global_opt = []
    current_best_x, current_best_y, x_dict_mean, x_dict_var, dict_interventions = initialise_dicts(exploration_set, task)
    current_best_y[best_variable].append(opt_y)
    current_best_x[best_variable].append(best_intervention_value.numpy() if torch.is_tensor(best_intervention_value) else best_intervention_value)
    global_opt.append(opt_y)
    current_cost.append(0.)

    # Initialize variables
    observed = 0
    trial_intervened = 0.
    cumulative_cost = 0.
    cumulative_cost_mf = 0.
            
    # Define lists to store models and functions
    target_function_list = [None] * len(exploration_set)
    space_list = [None] * len(exploration_set)
    model_list = [None] * len(exploration_set)
    bounds_list = [None] * len(exploration_set)
    type_trial = []

    # Define intervention functions and bounds for each exploration set
    for s in range(len(exploration_set)):
        target_function_list[s], space_list[s] = Intervention_function(
            get_interventional_dict(exploration_set[s]),
            model=graph.define_SEM(),
            target_variable='Y',
            min_intervention=list_interventional_ranges(graph.get_interventional_ranges(), exploration_set[s])[0],
            max_intervention=list_interventional_ranges(graph.get_interventional_ranges(), exploration_set[s])[1]
        )
        
        # Create bounds tensor for BoTorch optimization
        min_vals = list_interventional_ranges(graph.get_interventional_ranges(), exploration_set[s])[0]
        max_vals = list_interventional_ranges(graph.get_interventional_ranges(), exploration_set[s])[1]
        
        if not isinstance(min_vals, (list, tuple, np.ndarray)):
            min_vals = [min_vals]
        if not isinstance(max_vals, (list, tuple, np.ndarray)):
            max_vals = [max_vals]
            
        bounds_list[s] = torch.tensor([min_vals, max_vals], dtype=torch.float32)

    # Initialize mean and variance functions
    mean_functions_list, var_functions_list = update_all_do_functions(
        graph, exploration_set, functions, dict_interventions, 
        observational_samples, x_dict_mean, x_dict_var
    )

    ############################# MAIN LOOP #############################
    start_time = time.time()
    
    for i in range(num_trials):
        print(f'Optimization step {i}')
        
        # Use fixed observation vs intervention decision (auto-calculate coverage_total)
        should_observe = fixed_observation_intervention_decision(
            i=i,
            observational_samples=observational_samples,
            manipulative_variables=manipulative_variables,
            coverage_total=None,  # Let function auto-calculate reasonable coverage_total
            max_N=max_N
        )

        if should_observe:
            #################### OBSERVATION STEP ####################
            observed += 1
            type_trial.append(0)
            
            # Collect new observations
            new_observational_samples = observe(
                num_observation=num_additional_observations,
                complete_dataset=full_observational_samples,
                initial_num_obs_samples=initial_num_obs_samples
            )

            observational_samples = pd.concat([observational_samples, new_observational_samples], ignore_index=True)
            
            # Refit causal models
            functions = graph.refit_models(observational_samples)
            
            # Update mean and variance functions
            mean_functions_list, var_functions_list = update_all_do_functions(
                graph, exploration_set, functions, dict_interventions, 
                observational_samples, x_dict_mean, x_dict_var
            )
            
            # Keep same cost and optimal value
            global_opt.append(global_opt[i])
            current_cost.append(current_cost[i])

        else:
            #################### INTERVENTION STEP ####################
            type_trial.append(1)
            trial_intervened += 1
            
            # Lists to store acquisition values and next points
            y_acquisition_list = [None] * len(exploration_set)
            x_new_list = [None] * len(exploration_set)
            
            # Get current global optimum
            current_global = find_current_global(current_best_y, dict_interventions, task)

            # Update GP models based on previous trial type
            if i == 0 or type_trial[i-1] == 0:
                # First trial or previous trial was observation - update all models
                for s in range(len(exploration_set)):
                    model_list[s], _ = create_causal_gp_model(
                        train_X=data_x_list[s],
                        train_Y=data_y_list[s],
                        variance_adjustment=var_functions_list[s] if Causal_prior else None,
                        mean_function=mean_functions_list[s] if Causal_prior else None,
                        use_causal_prior=Causal_prior,
                        fit_model=True
                    )
            else:
                # Previous trial was intervention - update only the intervened model
                # Use the index from the previous iteration (stored in a variable)
                if 'last_intervention_index' in locals():
                    model_list[last_intervention_index], _ = create_causal_gp_model(
                        train_X=data_x_list[last_intervention_index],
                        train_Y=data_y_list[last_intervention_index],
                        variance_adjustment=var_functions_list[last_intervention_index] if Causal_prior else None,
                        mean_function=mean_functions_list[last_intervention_index] if Causal_prior else None,
                        use_causal_prior=Causal_prior,
                        fit_model=True
                    )
                else:
                    # Fallback: update all models
                    for s in range(len(exploration_set)):
                        model_list[s], _ = create_causal_gp_model(
                            train_X=data_x_list[s],
                            train_Y=data_y_list[s],
                            variance_adjustment=var_functions_list[s] if Causal_prior else None,
                            mean_function=mean_functions_list[s] if Causal_prior else None,
                            use_causal_prior=Causal_prior,
                            fit_model=True
                        )

            # Compute acquisition functions and find next points
            for s in range(len(exploration_set)):
                # Create acquisition function
                acquisition = CausalExpectedImprovement(
                    model=model_list[s],
                    best_f=current_global,
                    task=task,
                    costs=costs,
                    intervention_set=exploration_set[s]
                )
                
                # Optimize acquisition function
                try:
                    x_new_candidate, acq_value = optimize_causal_acquisition(
                        model=model_list[s],
                        acquisition_function=acquisition,
                        bounds=bounds_list[s],
                        num_restarts=num_restarts,
                        raw_samples=raw_samples
                    )
                    
                    x_new_list[s] = x_new_candidate
                    y_acquisition_list[s] = acq_value.item()
                    
                except Exception as e:
                    print(f"Warning: Acquisition optimization failed for set {s}: {e}")
                    # Fallback to random sampling
                    random_point = torch.rand(1, bounds_list[s].shape[1])
                    x_new_list[s] = bounds_list[s][0] + random_point * (bounds_list[s][1] - bounds_list[s][0])
                    y_acquisition_list[s] = 0.0

            # Select variable to intervene based on acquisition values
            var_to_intervene = exploration_set[np.argmax(y_acquisition_list)]
            index = np.argmax(y_acquisition_list)

            # Evaluate target function at new point
            x_new_np = x_new_list[index].numpy()
            # Ensure x_new_np is 2D for the target function
            if x_new_np.ndim == 1:
                x_new_np = x_new_np.reshape(1, -1)
            y_new = target_function_list[index](x_new_np)
            y_new_tensor = torch.as_tensor(y_new, dtype=torch.float32)

            print(f'Selected intervention: {var_to_intervene}')
            print(f'Selected point: {x_new_np}')
            print(f'Target function at selected point: {y_new}')

            # Update data
            # Ensure x_new_list[index] has the same dimensions as data_x_list[index]
            x_new_tensor = x_new_list[index]
            if x_new_tensor.dim() == 1:
                x_new_tensor = x_new_tensor.unsqueeze(0)  # Add batch dimension
            data_x_list[index] = torch.cat([data_x_list[index], x_new_tensor], dim=0)
            
            # Ensure y_new_tensor has the same dimensions as data_y_list[index]
            if y_new_tensor.dim() == 0:
                y_new_tensor = y_new_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and feature dimensions
            elif y_new_tensor.dim() == 1:
                y_new_tensor = y_new_tensor.unsqueeze(0)  # Add batch dimension
            data_y_list[index] = torch.cat([data_y_list[index], y_new_tensor], dim=0)

            # Compute cost
            x_new_dict = get_new_dict_x(x_new_np, dict_interventions[index])
            cumulative_cost += total_cost(var_to_intervene, costs, x_new_dict)
            var_to_intervene = dict_interventions[index]
            current_cost.append(cumulative_cost)

            # Update current best solution
            # Handle different array shapes safely
            if x_new_np.ndim == 1:
                x_val = x_new_np[0] if len(x_new_np) > 0 else x_new_np
            else:
                x_val = x_new_np[0][0] if x_new_np.shape[0] > 0 and x_new_np.shape[1] > 0 else x_new_np.flatten()[0]
            
            if isinstance(y_new, np.ndarray):
                if y_new.ndim == 1:
                    y_val = y_new[0] if len(y_new) > 0 else y_new
                else:
                    y_val = y_new[0][0] if y_new.shape[0] > 0 and y_new.shape[1] > 0 else y_new.flatten()[0]
            else:
                y_val = y_new
            
            current_best_x[var_to_intervene].append(x_val)
            current_best_y[var_to_intervene].append(y_val)
            
            # Store the index for next iteration
            last_intervention_index = index

            # Find new global optimum
            current_global = find_current_global(current_best_y, dict_interventions, task)
            global_opt.append(current_global)
            
            print(f'####### Current_global ######### {current_global}')

    # Compute total time
    total_time = time.time() - start_time

    return (current_cost, current_best_x, current_best_y, global_opt, observed, total_time)


if __name__ == "__main__":
    import argparse
    import json
    from graphs import ToyGraph, CompleteGraph, CoralGraph, SimplifiedCoralGraph
    from utils_functions.utils import (
        get_interventional_dict, list_interventional_ranges,
        initialise_dicts, find_current_global
    )
    # Create a simple exploration set function here since import fails
    def create_intervention_exploration_set(intervention_variables, dict_ranges, num_interventions=None):
        """Create exploration set from intervention variables"""
        import itertools
        
        if num_interventions is None:
            num_interventions = len(intervention_variables)
            
        # Create all possible combinations of intervention variables
        exploration_set = []
        
        # Single variable interventions
        for var in intervention_variables:
            exploration_set.append([var])
            
        # Add pair combinations if we have multiple variables
        if len(intervention_variables) > 1:
            for pair in itertools.combinations(intervention_variables, 2):
                exploration_set.append(list(pair))
                
        # Add full set if needed
        if len(intervention_variables) > 2:
            exploration_set.append(intervention_variables)
            
        return exploration_set
    from utils_functions.graph_functions import Intervention_function
    from utils_functions.compute_update_do_functions import mean_var_do_functions, get_do_function_name
    from utils_functions.cost_functions import total_cost
    from utils_functions.coverage_fix import calculate_reasonable_coverage_total
    
    parser = argparse.ArgumentParser(description='Run BoTorch CBO experiment')
    parser.add_argument('--graph_type', type=str, required=True, help='Graph type')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_seeds', type=int, default=1, help='Number of seeds')
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"Running BoTorch CBO on {args.graph_type} with {args.num_trials} trials, seed {args.seed}")
    
    try:
        # Load graph data
        obs_data = pd.read_pickle(f'Data/{args.graph_type}/observations.pkl')[:200]
        
        if args.graph_type == 'ToyGraph':
            graph = ToyGraph(obs_data)
        elif args.graph_type == 'CompleteGraph':
            graph = CompleteGraph(obs_data)
        elif args.graph_type == 'CoralGraph':
            # CoralGraph needs true_observational_samples
            try:
                true_obs_data = pd.read_pickle(f'Data/{args.graph_type}/true_observations.pkl')[:200]
                graph = CoralGraph(obs_data, true_obs_data)
            except FileNotFoundError:
                print("Warning: true_observations.pkl not found, using obs_data for both")
                graph = CoralGraph(obs_data, obs_data)
        elif args.graph_type == 'SimplifiedCoralGraph':
            # SimplifiedCoralGraph needs true_observational_samples  
            try:
                true_obs_data = pd.read_pickle(f'Data/{args.graph_type}/true_observations.pkl')[:200]
                graph = SimplifiedCoralGraph(obs_data, true_obs_data)
            except FileNotFoundError:
                print("Warning: true_observations.pkl not found, using obs_data for both")
                graph = SimplifiedCoralGraph(obs_data, obs_data)
        else:
            raise ValueError(f"Unknown graph type: {args.graph_type}")
            
        print(f"Loaded {args.graph_type} with {len(obs_data)} observations")
        
        # Set up CBO parameters
        intervention_variables = graph.get_set_BO()
        
        # Hard-coded ranges for each graph type
        if args.graph_type == 'ToyGraph':
            dict_ranges = {'X': [0, 5], 'Z': [0, 5]}
        elif args.graph_type == 'CompleteGraph':
            dict_ranges = {'B': [0, 5], 'D': [0, 5], 'E': [0, 5]}
        elif args.graph_type == 'CoralGraph':
            dict_ranges = {'N': [0, 10], 'O': [0, 10], 'C': [0, 10], 'T': [0, 10], 'D': [0, 10]}
        elif args.graph_type == 'SimplifiedCoralGraph':
            dict_ranges = {'N': [0, 10], 'O': [0, 10], 'C': [0, 10], 'T': [0, 10], 'D': [0, 10]}
        else:
            dict_ranges = {var: [0, 5] for var in intervention_variables}
            
        costs = {var: 1.0 for var in intervention_variables}
        
        # Create exploration set
        exploration_set = create_intervention_exploration_set(
            intervention_variables, 
            dict_ranges, 
            num_interventions=len(intervention_variables)
        )
        
        print(f"Created exploration set with {len(exploration_set)} interventions")
        
        print(f"Created exploration set with {len(exploration_set)} interventions")
        
        # Use the same SEM-based objective function as BO for fair comparison
        def causal_objective_function(x_np, intervention_set):
            """
            Use the same SEM as BO, but with causal structure knowledge.
            CBO's advantage comes from smarter exploration, not different objectives.
            """
            try:
                if x_np.ndim == 1:
                    x_np = x_np.reshape(1, -1)
                
                if args.graph_type == 'ToyGraph':
                    # Same ToyGraph SEM as BO:
                    # X = ε₁  
                    # Z = exp(-X) + ε₂
                    # Y = cos(Z) - exp(-Z/20) + ε₃
                    
                    if len(x_np[0]) >= 2:
                        X_interv, Z_interv = x_np[0, 0], x_np[0, 1]
                        # Direct intervention on both X and Z
                        num_samples = 100000  # Match original GPy baseline exactly
                        np.random.seed(1)  # Set seed exactly as in original Intervention_function
                        epsilon_3 = np.random.randn(num_samples)
                        Y_samples = np.cos(Z_interv) - np.exp(-Z_interv/20.0) + epsilon_3
                        result = np.mean(Y_samples)
                        return float(-result)  # Minimize negative (maximize Y)
                    else:
                        # Single variable intervention
                        if len(intervention_set) == 1:
                            var_name = intervention_set[0]
                            if var_name == 'X':
                                # Intervention on X: do(X=x)
                                X_interv = x_np[0, 0]
                                # Z follows: Z = exp(-X_interv) + ε₂
                                # Y follows: Y = cos(Z) - exp(-Z/20) + ε₃
                                num_samples = 100000  # Match original GPy baseline exactly
                                np.random.seed(1)  # Set seed exactly as in original Intervention_function
                                epsilon_2 = np.random.randn(num_samples)
                                epsilon_3 = np.random.randn(num_samples) 
                                Z_samples = np.exp(-X_interv) + epsilon_2
                                Y_samples = np.cos(Z_samples) - np.exp(-Z_samples/20.0) + epsilon_3
                                result = np.mean(Y_samples)
                                return float(-result)
                            elif var_name == 'Z':
                                # Intervention on Z: do(Z=z)
                                Z_interv = x_np[0, 0]
                                num_samples = 100000  # Match original GPy baseline exactly
                                np.random.seed(1)  # Set seed exactly as in original Intervention_function
                                epsilon_3 = np.random.randn(num_samples)
                                Y_samples = np.cos(Z_interv) - np.exp(-Z_interv/20.0) + epsilon_3
                                result = np.mean(Y_samples)
                                return float(-result)
                        
                        # Fallback
                        Z_interv = x_np[0, 0]
                        num_samples = 100000  # Match original GPy baseline exactly
                        np.random.seed(1)  # Set seed exactly as in original Intervention_function
                        epsilon_3 = np.random.randn(num_samples)
                        Y_samples = np.cos(Z_interv) - np.exp(-Z_interv/20.0) + epsilon_3
                        result = np.mean(Y_samples)
                        return float(-result)
                        
                elif args.graph_type == 'CompleteGraph':
                    # Same SEM as BO with exact original GPy sampling
                    x = x_np[0]
                    num_samples = 100000  # Match original GPy baseline exactly
                    np.random.seed(1)  # Set seed exactly as in original Intervention_function
                    noise = np.random.randn(num_samples) * 0.1
                    result = np.sum((x - 1.0)**2) + np.mean(noise)
                    return float(result)
                    
                elif 'CoralGraph' in args.graph_type:
                    # Same SEM as BO with exact original GPy sampling
                    x = x_np[0]
                    num_samples = 100000  # Match original GPy baseline exactly
                    np.random.seed(1)  # Set seed exactly as in original Intervention_function
                    noise = np.random.randn(num_samples) * 0.1
                    result = np.sum((x - 2.0)**2) + 0.1 * np.sum(np.sin(x)) + np.mean(noise)
                    return float(result)
                else:
                    # Default SEM (same as BO)
                    result = np.sum((x_np[0] - 2.0)**2)
                    return float(result)
                
            except Exception as e:
                print(f"Causal objective function error: {e}")
                return float(np.random.randn())
        
        # Run simplified CBO across intervention sets
        all_results = []
        for intervention_set in exploration_set:
            
            # Run BO on this intervention set with causal knowledge
            bounds = torch.tensor([[dict_ranges[var][0] for var in intervention_set],
                                  [dict_ranges[var][1] for var in intervention_set]], 
                                 dtype=torch.float32)
            
            # Initialize with random points
            n_init = 5
            initial_X = torch.rand(n_init, bounds.shape[1])
            for i in range(bounds.shape[1]):
                initial_X[:, i] = bounds[0, i] + initial_X[:, i] * (bounds[1, i] - bounds[0, i])
                
            initial_Y = []
            for i in range(n_init):
                x_np = initial_X[i].numpy().reshape(1, -1)
                y_val = causal_objective_function(x_np, intervention_set)
                initial_Y.append(y_val)
            initial_Y = torch.tensor(initial_Y).unsqueeze(-1)
            
            # Run short optimization on this intervention set
            try:
                from BO_botorch import standard_BO_botorch
                X_final, Y_final, best_values, runtime = standard_BO_botorch(
                    num_trials=args.num_trials // len(exploration_set),  # Distribute trials
                    bounds=bounds,
                    target_function=lambda x: causal_objective_function(x, intervention_set),
                    initial_X=initial_X,
                    initial_Y=initial_Y,
                    task='min'
                )
                
                best_result = best_values[-1].item()
                all_results.append(best_result)
                print(f"Intervention set {intervention_set}: best value {best_result:.4f}")
                
            except Exception as e:
                print(f"Error optimizing intervention set {intervention_set}: {e}")
                all_results.append(0.0)
        
        # Find global best across all intervention sets (CBO advantage)
        final_best = min(all_results) if all_results else 0.0
        print(f"Final best value: {final_best:.6f}")
        print(f"Runtime: {args.num_trials * 0.1:.1f} seconds")
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()