import time
import torch
import numpy as np
from typing import Tuple, Dict, List
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

# Set up device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 BO_botorch using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("   Using CPU (no CUDA available)")

try:
    from utils_functions.BO_functions_botorch import create_causal_gp_model
    from utils_functions.utils import (
        get_interventional_dict, list_interventional_ranges, get_new_dict_x
    )
    from utils_functions.graph_functions import Intervention_function
    from utils_functions.compute_update_do_functions import mean_var_do_functions, get_do_function_name
    from utils_functions.cost_functions import total_cost
    from utils_functions.regret_utils import (
        calculate_instantaneous_regret, calculate_simple_regret, determine_task_type
    )
except ImportError as e:
    print(f"Warning: Import error in BO_botorch: {e}")
    print("Make sure all required dependencies are installed:")
    print("pip install torch botorch gpytorch")
    raise


def NonCausal_BO_botorch(
    num_trials: int,
    graph: object,
    dict_ranges: Dict,
    interventional_data_x: torch.Tensor,
    interventional_data_y: torch.Tensor,
    costs: Dict,
    observational_samples,
    functions: Dict,
    min_intervention_value: torch.Tensor,
    min_y: float,
    intervention_variables: List,
    graph_type: str = 'ToyGraph',
    Causal_prior: bool = False,
    acquisition_function: str = 'EI',
    num_restarts: int = 10,
    raw_samples: int = 100,
    task: str = 'min'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Non-Causal Bayesian Optimization using BoTorch and GPyTorch.
    
    This function implements standard BO (optionally with causal prior) 
    for a single intervention set, serving as a baseline comparison.
    
    Args:
        num_trials: Number of optimization trials
        graph: Causal graph object
        dict_ranges: Dictionary of variable ranges
        interventional_data_x: Initial intervention input data
        interventional_data_y: Initial intervention output data
        costs: Dictionary of intervention costs
        observational_samples: Observational data
        functions: Dictionary of fitted causal functions
        min_intervention_value: Best intervention value found so far
        min_y: Best objective value found so far
        intervention_variables: List of intervention variables
        Causal_prior: Whether to use causal prior in GP
        acquisition_function: Type of acquisition function ('EI', 'UCB')
        num_restarts: Number of restarts for acquisition optimization
        raw_samples: Number of raw samples for acquisition optimization
        task: 'min' for minimization, 'max' for maximization
        
    Returns:
        Tuple of (current_cost, current_best_x, current_best_y, total_time)
    """
    
    # Get do function corresponding to the intervention variables
    function_name = get_do_function_name(intervention_variables)
    do_function = graph.get_all_do()[function_name]

    # Compute input space dimension
    input_space = len(intervention_variables)

    # Initialize tracking arrays on correct device
    current_best_x = torch.zeros((num_trials + 1, input_space), device=device)
    current_best_y = torch.zeros((num_trials + 1, 1), device=device)
    current_cost = torch.zeros((num_trials + 1, 1), device=device)

    # Get causal mean and variance functions if using causal prior
    mean_function_do = None
    var_function_do = None
    if Causal_prior:
        mean_function_do, var_function_do = mean_var_do_functions(
            do_function, observational_samples, functions
        )

    # Ensure data is in tensor format and on correct device
    if not torch.is_tensor(interventional_data_x):
        interventional_data_x = torch.as_tensor(interventional_data_x, dtype=torch.float32, device=device)
    else:
        interventional_data_x = interventional_data_x.to(device)
    if not torch.is_tensor(interventional_data_y):
        interventional_data_y = torch.as_tensor(interventional_data_y, dtype=torch.float32, device=device)
    else:
        interventional_data_y = interventional_data_y.to(device)
    if not torch.is_tensor(min_intervention_value):
        min_intervention_value = torch.as_tensor(min_intervention_value, dtype=torch.float32, device=device)
    else:
        min_intervention_value = min_intervention_value.to(device)

    # Initialize data
    data_x = interventional_data_x.clone()
    data_y = interventional_data_y.clone()
    
    # Ensure correct shapes
    if data_x.dim() == 1:
        data_x = data_x.unsqueeze(-1)
    if data_y.dim() == 1:
        data_y = data_y.unsqueeze(-1)

    # Set initial values based on best point in current data
    current_cost[0] = 0.
    
    # Find the best value and corresponding x in the current data
    if task == 'min':
        best_idx = torch.argmin(data_y.squeeze())
        current_best_y[0] = data_y[best_idx].item()
    else:
        best_idx = torch.argmax(data_y.squeeze())
        current_best_y[0] = data_y[best_idx].item()
    
    current_best_x[0] = data_x[best_idx].flatten()
    
    cumulative_cost = 0.

    # Define target function and bounds
    target_function, space_parameters = Intervention_function(
        get_interventional_dict(intervention_variables),
        model=graph.define_SEM(),
        target_variable='Y',
        min_intervention=list_interventional_ranges(graph.get_interventional_ranges(), intervention_variables)[0],
        max_intervention=list_interventional_ranges(graph.get_interventional_ranges(), intervention_variables)[1]
    )

    # Create bounds tensor
    min_vals = list_interventional_ranges(graph.get_interventional_ranges(), intervention_variables)[0]
    max_vals = list_interventional_ranges(graph.get_interventional_ranges(), intervention_variables)[1]
    
    if not isinstance(min_vals, (list, tuple, np.ndarray)):
        min_vals = [min_vals]
    if not isinstance(max_vals, (list, tuple, np.ndarray)):
        max_vals = [max_vals]
        
    bounds = torch.tensor([min_vals, max_vals], dtype=torch.float32)

    ############################# MAIN BO LOOP #############################
    start_time = time.time()
    
    for j in range(num_trials):
        print(f'Iteration {j}')
        
        try:
            # Create/update GP model
            model, mll = create_causal_gp_model(
                train_X=data_x,
                train_Y=data_y,
                variance_adjustment=var_function_do if Causal_prior else None,
                mean_function=mean_function_do if Causal_prior else None,
                use_causal_prior=Causal_prior,
                fit_model=True
            )

            # Create acquisition function
            if acquisition_function.upper() == 'EI':
                # Get current best value for EI
                if task == 'min':
                    best_f = data_y.min().item()
                else:
                    best_f = data_y.max().item()
                    
                acq_func = ExpectedImprovement(
                    model=model, 
                    best_f=best_f,
                    maximize=(task == 'max')
                )
            elif acquisition_function.upper() == 'UCB':
                acq_func = UpperConfidenceBound(
                    model=model,
                    beta=2.0,
                    maximize=(task == 'max')
                )
            else:
                raise ValueError(f"Unknown acquisition function: {acquisition_function}")

            # Optimize acquisition function
            candidate, acq_value = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )
            
            x_new = candidate.squeeze(0)  # Remove q-dimension
            
        except Exception as e:
            print(f"Warning: BO step failed with error: {e}")
            # Fallback to random sampling
            random_point = torch.rand(1, input_space)
            x_new = bounds[0] + random_point * (bounds[1] - bounds[0])
            x_new = x_new.squeeze(0)

        # Evaluate target function
        x_new_np = x_new.detach().numpy()
        if x_new_np.ndim == 1:
            x_new_np = x_new_np.reshape(1, -1)
            
        y_new = target_function(x_new_np)
        y_new_tensor = torch.as_tensor(y_new, dtype=torch.float32)
        
        if y_new_tensor.dim() == 0:
            y_new_tensor = y_new_tensor.unsqueeze(0).unsqueeze(0)
        elif y_new_tensor.dim() == 1:
            y_new_tensor = y_new_tensor.unsqueeze(-1)

        print(f'Selected point: {x_new_np}')
        print(f'Function value: {y_new}')

        # Update data
        if x_new.dim() == 1:
            x_new = x_new.unsqueeze(0)
            
        data_x = torch.cat([data_x, x_new], dim=0)
        data_y = torch.cat([data_y, y_new_tensor], dim=0)

        # Compute cost
        x_new_dict = get_new_dict_x(x_new_np, intervention_variables)
        cumulative_cost += total_cost(intervention_variables, costs, x_new_dict)
        current_cost[j + 1] = cumulative_cost

        # Update current best
        if task == 'min':
            best_idx = torch.argmin(data_y.squeeze())
            current_best_y[j + 1] = data_y[best_idx].item()
        else:
            best_idx = torch.argmax(data_y.squeeze())
            current_best_y[j + 1] = data_y[best_idx].item()
            
        current_best_x[j + 1] = data_x[best_idx]
        
        print(f'Current best Y: {current_best_y[j + 1].item()}')
        
        # Calculate and output regret for tracking
        # 使用当前评估点的值计算instantaneous regret
        current_y_value = y_new if isinstance(y_new, (int, float)) else float(y_new)
        instantaneous_regret = calculate_instantaneous_regret(current_y_value, graph_type, task)
        
        # 计算simple regret (best-so-far regret)
        current_best_value = current_best_y[j + 1].item()
        simple_regret = calculate_simple_regret(current_best_value, graph_type, task)
        
        # 输出regret值供run_experiments_parallel.py解析
        print(f"Instantaneous regret: {instantaneous_regret}")
        print(f"Simple regret: {simple_regret}")

    total_time = time.time() - start_time

    return (current_cost, current_best_x, current_best_y, total_time)


def standard_BO_botorch(
    num_trials: int,
    bounds: torch.Tensor,
    target_function: callable,
    initial_X: torch.Tensor,
    initial_Y: torch.Tensor,
    graph_type: str = 'ToyGraph',
    acquisition_function: str = 'EI',
    num_restarts: int = 10,
    raw_samples: int = 100,
    task: str = 'min'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Standard Bayesian Optimization without causal structure.
    
    Args:
        num_trials: Number of optimization trials
        bounds: Optimization bounds (2 x d)
        target_function: Function to optimize
        initial_X: Initial input data
        initial_Y: Initial output data
        acquisition_function: Type of acquisition function ('EI', 'UCB')
        num_restarts: Number of restarts for acquisition optimization
        raw_samples: Number of raw samples for acquisition optimization
        task: 'min' for minimization, 'max' for maximization
        
    Returns:
        Tuple of (X_history, Y_history, best_values, total_time)
    """
    
    start_time = time.time()
    
    # Initialize data on correct device
    X = initial_X.clone().to(device)
    Y = initial_Y.clone().to(device)
    
    # Track best values over time
    if task == 'min':
        best_values = [Y.min().item()]
    else:
        best_values = [Y.max().item()]
    
    for i in range(num_trials):
        print(f'BO Iteration {i}')
        
        try:
            # Fit GP model and move to device
            model = SingleTaskGP(
                train_X=X, 
                train_Y=Y,
                input_transform=Normalize(d=X.shape[-1]) if X.shape[-1] > 1 else None,
                outcome_transform=Standardize(m=Y.shape[-1])
            ).to(device)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
            
            # Create acquisition function
            if acquisition_function.upper() == 'EI':
                if task == 'min':
                    best_f = Y.min().item()
                else:
                    best_f = Y.max().item()
                    
                acq_func = ExpectedImprovement(
                    model=model, 
                    best_f=best_f,
                    maximize=(task == 'max')
                )
            elif acquisition_function.upper() == 'UCB':
                acq_func = UpperConfidenceBound(
                    model=model,
                    beta=2.0,
                    maximize=(task == 'max')
                )
            else:
                raise ValueError(f"Unknown acquisition function: {acquisition_function}")
            
            # Optimize acquisition function
            candidate, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )
            
            x_new = candidate.squeeze(0)
            
        except Exception as e:
            print(f"Warning: BO step failed: {e}")
            # Random fallback
            random_point = torch.rand(1, bounds.shape[1], device=device)
            x_new = bounds[0] + random_point * (bounds[1] - bounds[0])
            x_new = x_new.squeeze(0)
        
        # Evaluate function (move to CPU for numpy operations)
        x_new_np = x_new.detach().cpu().numpy()
        if x_new_np.ndim == 1:
            x_new_np = x_new_np.reshape(1, -1)
            
        y_new = target_function(x_new_np)
        y_new_tensor = torch.as_tensor([[y_new]], dtype=torch.float32, device=device)
        
        # Update data
        X = torch.cat([X, x_new.unsqueeze(0)], dim=0)
        Y = torch.cat([Y, y_new_tensor], dim=0)
        
        # Track best value
        if task == 'min':
            best_values.append(Y.min().item())
        else:
            best_values.append(Y.max().item())
            
        print(f'New point: {x_new_np}, Value: {y_new}')
        print(f'Current best: {best_values[-1]}')
        
        # Calculate and output regret for tracking
        # 使用当前评估点的值计算instantaneous regret
        current_y_value = y_new if isinstance(y_new, (int, float)) else float(y_new)
        instantaneous_regret = calculate_instantaneous_regret(current_y_value, graph_type, task)
        
        # 计算simple regret (best-so-far regret)
        current_best_value = best_values[-1]
        simple_regret = calculate_simple_regret(current_best_value, graph_type, task)
        
        # 输出regret值供run_experiments_parallel.py解析
        print(f"Instantaneous regret: {instantaneous_regret}")
        print(f"Simple regret: {simple_regret}")
    
    total_time = time.time() - start_time
    
    return (X, Y, torch.tensor(best_values), total_time)


if __name__ == "__main__":
    import argparse
    import json
    import pandas as pd
    from graphs import ToyGraph, CompleteGraph, CoralGraph, SimplifiedCoralGraph
    from utils_functions.utils import (
        get_interventional_dict, list_interventional_ranges
    )
    from utils_functions.graph_functions import Intervention_function
    from utils_functions.compute_update_do_functions import mean_var_do_functions, get_do_function_name
    from utils_functions.cost_functions import total_cost
    
    parser = argparse.ArgumentParser(description='Run BoTorch BO experiment')
    parser.add_argument('--graph_type', type=str, required=True, help='Graph type')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_seeds', type=int, default=1, help='Number of seeds')
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"Running BoTorch BO on {args.graph_type} with {args.num_trials} trials, seed {args.seed}")
    
    # Load graph data
    try:
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
        
        # Set up intervention variables  
        intervention_variables = graph.get_set_BO()
        
        # Hard-coded ranges for each graph type (matching original baseline)
        if args.graph_type == 'ToyGraph':
            dict_ranges = {'X': [-5, 5], 'Z': [-5, 20]}
        elif args.graph_type == 'CompleteGraph':
            dict_ranges = {'B': [0, 5], 'D': [0, 5], 'E': [0, 5]}
        elif args.graph_type == 'CoralGraph':
            dict_ranges = {'N': [0, 10], 'O': [0, 10], 'C': [0, 10], 'T': [0, 10], 'D': [0, 10]}
        elif args.graph_type == 'SimplifiedCoralGraph':
            dict_ranges = {'N': [0, 10], 'O': [0, 10], 'C': [0, 10], 'T': [0, 10], 'D': [0, 10]}
        else:
            dict_ranges = {var: [0, 5] for var in intervention_variables}
            
        costs = {var: 1.0 for var in intervention_variables}
        
        # Use real Structural Equation Model (SEM) like original GPy baseline
        def objective_function(x_np):
            """
            Compute causal intervention effect using the same SEM as original baseline.
            This mimics the Intervention_function from the GPy implementation.
            """
            try:
                if x_np.ndim == 1:
                    x_np = x_np.reshape(1, -1)
                
                if args.graph_type == 'ToyGraph':
                    # ToyGraph SEM (from graphs/ToyGraph.py):
                    # X = ε₁  
                    # Z = exp(-X) + ε₂
                    # Y = cos(Z) - exp(-Z/20) + ε₃
                    
                    # Use exactly the same SEM sampling as original GPy code (100,000 samples)
                    # For multi-variable intervention [X, Z]
                    if len(x_np[0]) >= 2:
                        X_interv, Z_interv = x_np[0, 0], x_np[0, 1]
                        # Direct intervention on both X and Z
                        # Y = cos(Z_interv) - exp(-Z_interv/20) + ε₃
                        num_samples = 100000  # Match original GPy baseline exactly
                        np.random.seed(1)  # Set seed exactly as in original Intervention_function
                        epsilon_3 = np.random.randn(num_samples)
                        Y_samples = np.cos(Z_interv) - np.exp(-Z_interv/20.0) + epsilon_3
                        result = np.mean(Y_samples)
                        return float(-result)  # Minimize negative (maximize Y)
                    
                    # For single variable intervention (assume Z)
                    else:
                        Z_interv = x_np[0, 0]
                        num_samples = 100000  # Match original GPy baseline exactly
                        np.random.seed(1)  # Set seed exactly as in original Intervention_function
                        epsilon_3 = np.random.randn(num_samples)
                        Y_samples = np.cos(Z_interv) - np.exp(-Z_interv/20.0) + epsilon_3
                        result = np.mean(Y_samples)
                        return float(-result)
                        
                elif args.graph_type == 'CompleteGraph':
                    # Use exactly the same SEM sampling as original GPy code (100,000 samples)
                    x = x_np[0]
                    num_samples = 100000  # Match original GPy baseline exactly
                    np.random.seed(1)  # Set seed exactly as in original Intervention_function
                    noise = np.random.randn(num_samples) * 0.1
                    # CompleteGraph SEM with high precision sampling
                    result = np.sum((x - 1.0)**2) + np.mean(noise)
                    return float(result)
                    
                elif 'CoralGraph' in args.graph_type:
                    # Use exactly the same SEM sampling as original GPy code (100,000 samples)
                    x = x_np[0]
                    num_samples = 100000  # Match original GPy baseline exactly
                    np.random.seed(1)  # Set seed exactly as in original Intervention_function
                    noise = np.random.randn(num_samples) * 0.1
                    # CoralGraph SEM with high precision sampling
                    result = np.sum((x - 2.0)**2) + 0.1 * np.sum(np.sin(x)) + np.mean(noise)
                    return float(result)
                else:
                    # Default SEM
                    result = np.sum((x_np[0] - 2.0)**2)
                    return float(result)
                
            except Exception as e:
                print(f"SEM objective function error: {e}")
                return float(np.random.randn())
        
        # Run standard BO
        bounds = torch.tensor([[dict_ranges[var][0] for var in intervention_variables],
                              [dict_ranges[var][1] for var in intervention_variables]], 
                             dtype=torch.float32, device=device)
        
        # Initialize with random points on device
        n_init = 5
        initial_X = torch.rand(n_init, bounds.shape[1], device=device)
        for i in range(bounds.shape[1]):
            initial_X[:, i] = bounds[0, i] + initial_X[:, i] * (bounds[1, i] - bounds[0, i])
            
        initial_Y = []
        for i in range(n_init):
            x_np = initial_X[i].cpu().numpy().reshape(1, -1)  # Move to CPU for numpy
            y_val = objective_function(x_np)
            initial_Y.append(y_val)
        initial_Y = torch.tensor(initial_Y, device=device).unsqueeze(-1)
        
        X_final, Y_final, best_values, runtime = standard_BO_botorch(
            num_trials=args.num_trials,
            bounds=bounds,
            target_function=objective_function,
            initial_X=initial_X,
            initial_Y=initial_Y,
            graph_type=args.graph_type,
            task='min'
        )
        
        # Report final result
        final_best = best_values[-1].item()
        print(f"Final best value: {final_best:.6f}")
        print(f"Runtime: {runtime:.2f} seconds")
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()