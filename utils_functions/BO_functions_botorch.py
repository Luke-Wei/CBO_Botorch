import torch
import gpytorch
import numpy as np
from typing import Optional, Callable, Tuple, Union
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

# Global device setting for consistency
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    from .causal_kernels_botorch import CausalRBFKernel, StandardRBFKernel
    from .causal_acquisition_functions_botorch import CausalExpectedImprovement
except ImportError:
    # Fallback for different import contexts
    from causal_kernels_botorch import CausalRBFKernel, StandardRBFKernel
    from causal_acquisition_functions_botorch import CausalExpectedImprovement


class CausalGP(SingleTaskGP):
    """
    Causal Gaussian Process model that incorporates causal structure through
    a causal kernel and mean function.
    """
    
    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        variance_adjustment: Optional[Callable] = None,
        mean_function: Optional[Callable] = None,
        likelihood: Optional[GaussianLikelihood] = None,
        input_transform: Optional[object] = None,
        outcome_transform: Optional[object] = None,
    ):
        """
        Initialize Causal GP.
        
        Args:
            train_X: Training inputs (n x d)
            train_Y: Training outputs (n x 1)
            variance_adjustment: Function for causal variance adjustment
            mean_function: Causal mean function from do-calculus
            likelihood: Gaussian likelihood
            input_transform: Input transformation (e.g., normalization)
            outcome_transform: Outcome transformation (e.g., standardization)
        """
        # Set up likelihood
        if likelihood is None:
            likelihood = GaussianLikelihood(
                noise_constraint=Interval(1e-6, 1e-2)
            )
        
        # Set up input and outcome transforms
        if input_transform is None and train_X.shape[-1] > 1:
            input_transform = Normalize(d=train_X.shape[-1])
        
        if outcome_transform is None:
            outcome_transform = Standardize(m=train_Y.shape[-1])
        
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        
        self.variance_adjustment = variance_adjustment
        self.mean_function = mean_function
        
        # Set up mean and covariance
        self._setup_model()
    
    def _setup_model(self):
        """Set up the mean and covariance modules."""
        # Mean module
        if self.mean_function is not None:
            # Use causal mean function
            self.mean_module = CausalMean(self.mean_function)
        else:
            self.mean_module = ConstantMean()
        
        # Covariance module
        if self.variance_adjustment is not None:
            # Use causal kernel
            base_kernel = CausalRBFKernel(
                variance_adjustment=self.variance_adjustment,
                ard_num_dims=self.train_inputs[0].shape[-1]
            )
        else:
            # Use standard RBF kernel
            base_kernel = RBFKernel(ard_num_dims=self.train_inputs[0].shape[-1])
        
        self.covar_module = ScaleKernel(base_kernel)


class CausalMean(gpytorch.means.Mean):
    """
    Mean module that uses a causal mean function from do-calculus.
    """
    
    def __init__(self, mean_function: Callable):
        super().__init__()
        self.mean_function = mean_function
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the causal mean function.
        
        Args:
            x: Input points
            
        Returns:
            Mean values
        """
        # Convert to numpy for the causal function, then back to tensor
        if torch.is_tensor(x):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.array(x)
        
        try:
            mean_vals = self.mean_function(x_np)
            if not torch.is_tensor(mean_vals):
                mean_vals = torch.as_tensor(mean_vals, dtype=x.dtype, device=x.device)
            
            # Ensure correct shape
            # Ensure correct shape for GPyTorch
            if mean_vals.dim() == 0:
                mean_vals = mean_vals.expand(x.shape[0])
            elif mean_vals.dim() == 1 and mean_vals.shape[0] != x.shape[0]:
                mean_vals = mean_vals.expand(x.shape[0])
            elif mean_vals.dim() == 2:
                mean_vals = mean_vals.squeeze(-1)
                
            return mean_vals  # Return (n,) shape
            
        except Exception as e:
            # Fallback to zero mean
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)


def create_causal_gp_model(
    train_X: Union[np.ndarray, torch.Tensor],
    train_Y: Union[np.ndarray, torch.Tensor],
    variance_adjustment: Optional[Callable] = None,
    mean_function: Optional[Callable] = None,
    use_causal_prior: bool = True,
    fit_model: bool = True,
) -> Tuple[CausalGP, ExactMarginalLogLikelihood]:
    """
    Create and optionally fit a Causal GP model.
    
    Args:
        train_X: Training inputs
        train_Y: Training outputs  
        variance_adjustment: Causal variance adjustment function
        mean_function: Causal mean function
        use_causal_prior: Whether to use causal prior
        fit_model: Whether to fit the model
        
    Returns:
        Tuple of (model, mll)
    """
    # Convert to tensors and move to device
    if not torch.is_tensor(train_X):
        train_X = torch.as_tensor(train_X, dtype=torch.float32, device=device)
    else:
        train_X = train_X.to(device)
    if not torch.is_tensor(train_Y):
        train_Y = torch.as_tensor(train_Y, dtype=torch.float32, device=device)
    else:
        train_Y = train_Y.to(device)
    
    # Ensure correct shapes
    if train_X.dim() == 1:
        train_X = train_X.unsqueeze(-1)
    if train_Y.dim() == 1:
        train_Y = train_Y.unsqueeze(-1)
    
    # Create model
    if use_causal_prior and (variance_adjustment is not None or mean_function is not None):
        model = CausalGP(
            train_X=train_X,
            train_Y=train_Y,
            variance_adjustment=variance_adjustment,
            mean_function=mean_function,
        ).to(device)
    else:
        # Standard GP without causal prior
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=Normalize(d=train_X.shape[-1]) if train_X.shape[-1] > 1 else None,
            outcome_transform=Standardize(m=train_Y.shape[-1]),
        ).to(device)
    
    # Set up marginal log likelihood
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
    # Fit model
    if fit_model:
        model.train()
        try:
            fit_gpytorch_model(mll)
        except Exception as e:
            print(f"Warning: Model fitting failed with error: {e}")
            # Continue with unfitted model
    
    model.eval()
    return model, mll


def optimize_causal_acquisition(
    model: CausalGP,
    acquisition_function: CausalExpectedImprovement,
    bounds: torch.Tensor,
    num_restarts: int = 10,
    raw_samples: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimize the causal acquisition function.
    
    Args:
        model: Fitted GP model
        acquisition_function: Causal acquisition function
        bounds: Optimization bounds (2 x d)
        num_restarts: Number of restarts for optimization
        raw_samples: Number of raw samples for initialization
        
    Returns:
        Tuple of (best_point, best_value)
    """
    # Ensure model is in eval mode
    model.eval()
    
    try:
        candidate, acq_value = optimize_acqf(
            acq_function=acquisition_function,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        
        return candidate.squeeze(0), acq_value
        
    except Exception as e:
        print(f"Warning: Acquisition optimization failed: {e}")
        # Fallback to random sampling
        return _random_sample_within_bounds(bounds), torch.tensor(0.0)


def _random_sample_within_bounds(bounds: torch.Tensor) -> torch.Tensor:
    """
    Sample a random point within the given bounds.
    
    Args:
        bounds: Bounds tensor (2 x d)
        
    Returns:
        Random point within bounds (1 x d)
    """
    d = bounds.shape[1]
    lower_bounds = bounds[0]
    upper_bounds = bounds[1]
    
    random_point = torch.rand(1, d)
    scaled_point = lower_bounds + random_point * (upper_bounds - lower_bounds)
    
    return scaled_point


def update_causal_gp_model(
    model: CausalGP,
    new_X: Union[np.ndarray, torch.Tensor],
    new_Y: Union[np.ndarray, torch.Tensor],
    variance_adjustment: Optional[Callable] = None,
    mean_function: Optional[Callable] = None,
    refit: bool = True,
) -> Tuple[CausalGP, ExactMarginalLogLikelihood]:
    """
    Update a Causal GP model with new data.
    
    Args:
        model: Existing GP model
        new_X: New input data
        new_Y: New output data
        variance_adjustment: Updated variance adjustment function
        mean_function: Updated mean function
        refit: Whether to refit the model
        
    Returns:
        Updated model and mll
    """
    # Convert to tensors
    if not torch.is_tensor(new_X):
        new_X = torch.as_tensor(new_X, dtype=torch.float32)
    if not torch.is_tensor(new_Y):
        new_Y = torch.as_tensor(new_Y, dtype=torch.float32)
    
    # Ensure correct shapes
    if new_X.dim() == 1:
        new_X = new_X.unsqueeze(-1)
    if new_Y.dim() == 1:
        new_Y = new_Y.unsqueeze(-1)
    
    # Get existing data
    train_X = model.train_inputs[0]
    train_Y = model.train_targets.unsqueeze(-1)
    
    # Combine old and new data
    updated_X = torch.cat([train_X, new_X], dim=0)
    updated_Y = torch.cat([train_Y, new_Y], dim=0)
    
    # Create new model with updated data
    updated_model, mll = create_causal_gp_model(
        train_X=updated_X,
        train_Y=updated_Y,
        variance_adjustment=variance_adjustment or model.variance_adjustment,
        mean_function=mean_function or model.mean_function,
        use_causal_prior=hasattr(model, 'variance_adjustment'),
        fit_model=refit,
    )
    
    return updated_model, mll


def define_initial_data_BO_botorch(
    interventional_data: list,
    num_interventions: int,
    intervention_variables: list,
    name_index: int,
    task: str = 'min'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Prepare initial data for BoTorch BO.
    
    Args:
        interventional_data: List of interventional datasets
        num_interventions: Number of initial interventions
        intervention_variables: List of intervention variables
        name_index: Random seed index
        task: 'min' or 'max'
        
    Returns:
        Tuple of (data_x, data_y, min_intervention_value, min_y)
    """
    # Handle ToyGraph data format: [num_vars, var_name, data_x, data_y]
    # Find the data for the specific intervention variables
    found_data = False
    for intervention_set in interventional_data:
        if len(intervention_set) >= 4:
            var_name = intervention_set[1]
            if (isinstance(intervention_variables, list) and len(intervention_variables) == 1 and 
                intervention_variables[0] == var_name) or intervention_variables == var_name:
                data_x = intervention_set[2].copy()
                data_y = intervention_set[3].copy()
                found_data = True
                break
    
    if not found_data:
        # Fallback: use first available dataset or create dummy data
        if len(interventional_data) > 0 and len(interventional_data[0]) >= 4:
            data_x = interventional_data[0][2].copy()
            data_y = interventional_data[0][3].copy()
        else:
            # Create dummy data for testing
            data_x = np.random.randn(20, len(intervention_variables))
            data_y = np.random.randn(20, 1)
    all_data = np.concatenate((data_x, data_y), axis=1)

    # Set random seed for reproducibility
    state = np.random.get_state()
    np.random.seed(name_index)
    np.random.shuffle(all_data)
    np.random.set_state(state)

    data_x = all_data[:num_interventions, :len(intervention_variables)]
    data_y = all_data[:num_interventions, len(intervention_variables):]

    # Find optimal value
    if task == 'min':
        min_y = np.min(data_y)
        min_idx = np.argmin(data_y)
    else:
        min_y = np.max(data_y)
        min_idx = np.argmax(data_y)
    
    min_intervention_value = data_x[min_idx:min_idx+1]

    # Convert to tensors
    data_x_tensor = torch.as_tensor(data_x, dtype=torch.float32)
    data_y_tensor = torch.as_tensor(data_y, dtype=torch.float32)
    min_intervention_tensor = torch.as_tensor(min_intervention_value, dtype=torch.float32)

    return data_x_tensor, data_y_tensor, min_intervention_tensor, float(min_y)