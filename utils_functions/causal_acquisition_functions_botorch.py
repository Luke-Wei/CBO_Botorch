import torch
import numpy as np
from typing import Optional, Union, Callable
from torch import Tensor
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch.distributions import Normal


class CausalExpectedImprovement(AcquisitionFunction):
    """
    Causal Expected Improvement acquisition function.
    
    This acquisition function computes the expected improvement over the current best 
    observed value, adapted for causal optimization where we consider the cost of 
    interventions and the causal structure.
    
    The acquisition function is defined as:
    EI(x) = σ(x) * [u * Φ(u) + φ(u)]
    
    where u = (f_min - μ(x)) / σ(x), Φ is the CDF and φ is the PDF of standard normal.
    For maximization, we negate the improvement.
    """
    
    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        task: str = 'min',
        maximize: bool = True,
        jitter: float = 0.0,
        costs: Optional[dict] = None,
        intervention_set: Optional[list] = None,
    ):
        """
        Initialize Causal Expected Improvement.
        
        Args:
            model: A fitted GP model
            best_f: Current best function value (global optimum)
            task: 'min' for minimization, 'max' for maximization  
            maximize: If True, maximize the acquisition (standard for BoTorch)
            jitter: Small value for numerical stability
            costs: Dictionary of intervention costs
            intervention_set: Set of variables being intervened on
        """
        super().__init__(model=model)
        
        self.register_buffer("best_f", torch.as_tensor(best_f, dtype=torch.float32))
        self.task = task
        self.maximize = maximize
        self.jitter = jitter
        self.costs = costs or {}
        self.intervention_set = intervention_set or []
        
        # Adjust sign based on task and maximize setting
        # For minimization task with maximize=True: need to flip sign
        # For maximization task with maximize=True: keep sign
        self.sign = 1.0
        if task == 'min' and maximize:
            self.sign = 1.0  # Standard EI for minimization
        elif task == 'max' and maximize:
            self.sign = -1.0  # Flip for maximization
        elif task == 'min' and not maximize:
            self.sign = -1.0  # Flip for minimization with maximize=False
        else:  # task == 'max' and not maximize
            self.sign = 1.0
    
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate the acquisition function at candidate points X.
        
        Args:
            X: Candidate points (batch_shape x q x d)
            
        Returns:
            Expected improvement values (batch_shape)
        """
        # X is already properly shaped by t_batch_mode_transform
        # Shape should be (batch_shape x 1 x d) after transform
        
        # Get model predictions
        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(-1).squeeze(-1)  # Remove both output and q dimensions
        variance = posterior.variance.squeeze(-1).squeeze(-1)  # Remove both output and q dimensions
        
        # Add jitter for numerical stability
        mean = mean + self.jitter
        
        # Compute standard deviation
        sigma = torch.sqrt(variance.clamp_min(1e-12))
        
        # Compute improvement
        if self.task == 'min':
            # For minimization: improvement when f < best_f
            u = (self.best_f - mean) / sigma
        else:
            # For maximization: improvement when f > best_f  
            u = (mean - self.best_f) / sigma
        
        # Standard normal distribution
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        cdf = normal.cdf(u)
        pdf = torch.exp(normal.log_prob(u))
        
        # Expected improvement formula
        ei = sigma * (u * cdf + pdf)
        
        # Apply sign correction for BoTorch convention
        ei = self.sign * ei
        
        # Apply cost adjustment if provided
        if self.costs and self.intervention_set:
            cost_factor = self._compute_cost_factor(X)
            ei = ei / cost_factor
        
        # Ensure the output shape matches the batch shape
        # ei should be (batch_shape,) not (batch_shape, q) or (batch_shape, d)
        if ei.dim() > 1:
            ei = ei.squeeze(-1)  # Remove trailing dimensions if any
        
        return ei
    
    def _compute_cost_factor(self, X: Tensor) -> Tensor:
        """
        Compute cost factor for intervention.
        
        Args:
            X: Candidate points
            
        Returns:
            Cost factors
        """
        # Simple cost implementation - can be extended
        base_cost = 1.0
        
        try:
            if isinstance(self.intervention_set, str):
                cost_val = self.costs.get(self.intervention_set, 1.0)
                # If cost is a function, evaluate it with a dummy value
                if callable(cost_val):
                    intervention_cost = 1.0  # Use default instead of evaluating function
                else:
                    intervention_cost = cost_val
            else:
                # Multiple interventions - sum costs
                intervention_cost = 0.0
                for var in self.intervention_set:
                    cost_val = self.costs.get(var, 1.0)
                    if callable(cost_val):
                        intervention_cost += 1.0  # Use default for functions
                    else:
                        intervention_cost += cost_val
            
            total_cost = base_cost + intervention_cost
        except:
            # Fallback to simple cost
            total_cost = 2.0
        
        # Return tensor with shape (batch_shape,) matching the expected output shape
        cost_tensor = torch.full_like(X[..., 0], total_cost)
        # Ensure it's 1D for batch shape
        if cost_tensor.dim() > 1:
            cost_tensor = cost_tensor.squeeze(-1)
        return cost_tensor


class CausalUpperConfidenceBound(AcquisitionFunction):
    """
    Causal Upper Confidence Bound acquisition function.
    
    UCB(x) = μ(x) + β * σ(x)
    
    For minimization, we use: -μ(x) + β * σ(x)
    """
    
    def __init__(
        self,
        model: Model,
        beta: Union[float, Tensor] = 2.0,
        task: str = 'min',
        maximize: bool = True,
        costs: Optional[dict] = None,
        intervention_set: Optional[list] = None,
    ):
        """
        Initialize Causal UCB.
        
        Args:
            model: A fitted GP model
            beta: Exploration parameter
            task: 'min' for minimization, 'max' for maximization
            maximize: If True, maximize the acquisition
            costs: Dictionary of intervention costs
            intervention_set: Set of variables being intervened on
        """
        super().__init__(model=model)
        
        self.register_buffer("beta", torch.as_tensor(beta, dtype=torch.float32))
        self.task = task
        self.maximize = maximize
        self.costs = costs or {}
        self.intervention_set = intervention_set or []
    
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate UCB at candidate points X.
        
        Args:
            X: Candidate points (batch_shape x q x d)
            
        Returns:
            UCB values (batch_shape)
        """
        # X is already properly shaped by t_batch_mode_transform
        
        # Get model predictions
        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(-1).squeeze(-1)  # Remove both output and q dimensions
        variance = posterior.variance.squeeze(-1).squeeze(-1)  # Remove both output and q dimensions
        sigma = torch.sqrt(variance.clamp_min(1e-12))
        
        # Compute UCB
        if self.task == 'min':
            # For minimization: lower confidence bound (negative UCB)
            ucb = -mean + self.beta * sigma
        else:
            # For maximization: upper confidence bound
            ucb = mean + self.beta * sigma
        
        # Adjust for BoTorch maximize convention
        if not self.maximize:
            ucb = -ucb
        
        # Apply cost adjustment if provided
        if self.costs and self.intervention_set:
            cost_factor = self._compute_cost_factor(X)
            ucb = ucb / cost_factor
        
        # Ensure the output shape matches the batch shape
        # ucb should be (batch_shape,) not (batch_shape, q) or (batch_shape, d)
        if ucb.dim() > 1:
            ucb = ucb.squeeze(-1)  # Remove trailing dimensions if any
        
        return ucb
    
    def _compute_cost_factor(self, X: Tensor) -> Tensor:
        """Compute cost factor for intervention."""
        base_cost = 1.0
        
        try:
            if isinstance(self.intervention_set, str):
                cost_val = self.costs.get(self.intervention_set, 1.0)
                if callable(cost_val):
                    intervention_cost = 1.0  # Use default for functions
                else:
                    intervention_cost = cost_val
            else:
                intervention_cost = 0.0
                for var in self.intervention_set:
                    cost_val = self.costs.get(var, 1.0)
                    if callable(cost_val):
                        intervention_cost += 1.0  # Use default for functions
                    else:
                        intervention_cost += cost_val
            
            total_cost = base_cost + intervention_cost
        except:
            total_cost = 2.0
        
        # Return tensor with shape (batch_shape,) matching the expected output shape
        cost_tensor = torch.full_like(X[..., 0], total_cost)
        # Ensure it's 1D for batch shape
        if cost_tensor.dim() > 1:
            cost_tensor = cost_tensor.squeeze(-1)
        return cost_tensor


def get_causal_acquisition_function(
    acq_func_name: str,
    model: Model,
    best_f: Union[float, Tensor],
    task: str = 'min',
    costs: Optional[dict] = None,
    intervention_set: Optional[list] = None,
    **kwargs
) -> AcquisitionFunction:
    """
    Factory function to create causal acquisition functions.
    
    Args:
        acq_func_name: Name of acquisition function ('EI', 'UCB')
        model: Fitted GP model
        best_f: Current best function value
        task: 'min' or 'max'
        costs: Intervention costs
        intervention_set: Variables being intervened on
        **kwargs: Additional arguments for acquisition function
        
    Returns:
        Causal acquisition function
    """
    if acq_func_name.upper() == 'EI':
        return CausalExpectedImprovement(
            model=model,
            best_f=best_f,
            task=task,
            costs=costs,
            intervention_set=intervention_set,
            **kwargs
        )
    elif acq_func_name.upper() == 'UCB':
        return CausalUpperConfidenceBound(
            model=model,
            task=task,
            costs=costs,
            intervention_set=intervention_set,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown acquisition function: {acq_func_name}")