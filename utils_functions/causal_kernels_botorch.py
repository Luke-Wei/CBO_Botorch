import torch
import numpy as np
from gpytorch.kernels import RBFKernel, Kernel
from gpytorch.constraints import Positive
from gpytorch.priors import SmoothedBoxPrior
from typing import Optional, Callable
from torch import Tensor


class CausalRBFKernel(Kernel):
    """
    Causal RBF kernel that incorporates variance adjustment based on causal structure.
    
    This kernel extends the standard RBF kernel by adding a causal variance adjustment term
    based on the do-calculus. The kernel has the form:
    
    K(x, x') = σ²exp(-0.5 * ||x-x'||²/l²) + √var_adj(x) × √var_adj(x')ᵀ
    
    where var_adj is the variance adjustment function from causal inference.
    """
    
    has_lengthscale = True
    
    def __init__(
        self,
        variance_adjustment: Callable[[Tensor], Tensor],
        ard_num_dims: Optional[int] = None,
        batch_shape: torch.Size = torch.Size(),
        active_dims: Optional[tuple] = None,
        lengthscale_prior: Optional[object] = None,
        lengthscale_constraint: Optional[object] = None,
        eps: float = 1e-6,
        **kwargs
    ):
        """
        Initialize the Causal RBF kernel.
        
        Args:
            variance_adjustment: Function that computes variance adjustment for causal prior
            ard_num_dims: Number of dimensions for ARD (automatic relevance determination)
            batch_shape: Batch shape for the kernel
            active_dims: Active dimensions for the kernel
            lengthscale_prior: Prior for lengthscale parameter
            lengthscale_constraint: Constraint for lengthscale parameter
            eps: Small value for numerical stability
        """
        super().__init__(
            batch_shape=batch_shape,
            active_dims=active_dims,
            ard_num_dims=ard_num_dims,
            **kwargs
        )
        
        self.variance_adjustment = variance_adjustment
        self.eps = eps
        
        # Initialize lengthscale parameter
        lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(torch.zeros((*self.batch_shape, 1, lengthscale_num_dims)))
        )
        
        if lengthscale_constraint is None:
            lengthscale_constraint = Positive()
        
        if lengthscale_prior is not None:
            self.register_prior(
                "lengthscale_prior",
                lengthscale_prior,
                lambda m: m.lengthscale,
                lambda m, v: m._set_lengthscale(v),
            )
        
        self.register_constraint("raw_lengthscale", lengthscale_constraint)
        
        # Initialize outputscale (variance) parameter
        self.register_parameter(
            name="raw_outputscale",
            parameter=torch.nn.Parameter(torch.zeros(self.batch_shape))
        )
        self.register_constraint("raw_outputscale", Positive())
    
    @property
    def lengthscale(self) -> Tensor:
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)
    
    @lengthscale.setter
    def lengthscale(self, value: Tensor):
        self._set_lengthscale(value)
    
    def _set_lengthscale(self, value: Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value))
    
    @property
    def outputscale(self) -> Tensor:
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)
    
    @outputscale.setter
    def outputscale(self, value: Tensor):
        self._set_outputscale(value)
    
    def _set_outputscale(self, value: Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))
    
    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params
    ) -> Tensor:
        """
        Compute the kernel matrix between x1 and x2.
        
        Args:
            x1: First set of inputs (n x d)
            x2: Second set of inputs (m x d)  
            diag: If True, return only diagonal elements
            last_dim_is_batch: Whether last dimension is batch dimension
            
        Returns:
            Kernel matrix (n x m) or diagonal (n,) if diag=True
        """
        # Ensure inputs are tensors
        if not torch.is_tensor(x1):
            x1 = torch.as_tensor(x1, dtype=torch.float32)
        if not torch.is_tensor(x2):
            x2 = torch.as_tensor(x2, dtype=torch.float32)
        
        # Compute standard RBF kernel
        # Distance computation
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)
            
        # Scale by lengthscale
        x1_scaled = x1.div(self.lengthscale)
        x2_scaled = x2.div(self.lengthscale)
        
        # Compute squared distances
        if diag:
            # For diagonal case, x1 and x2 should have same shape
            sq_dist = ((x1_scaled - x2_scaled) ** 2).sum(dim=-1)
        else:
            # Full kernel matrix
            sq_dist = torch.cdist(x1_scaled, x2_scaled, p=2) ** 2
        
        # Standard RBF component
        rbf_kernel = self.outputscale * torch.exp(-0.5 * sq_dist)
        
        # Compute causal variance adjustment
        try:
            var_adj_x1 = self.variance_adjustment(x1)
            var_adj_x2 = self.variance_adjustment(x2)
            
            # Ensure variance adjustments are tensors and have correct shape
            if not torch.is_tensor(var_adj_x1):
                var_adj_x1 = torch.as_tensor(var_adj_x1, dtype=torch.float32)
            if not torch.is_tensor(var_adj_x2):
                var_adj_x2 = torch.as_tensor(var_adj_x2, dtype=torch.float32)
            
            # Handle scalar case
            if var_adj_x1.dim() == 0:
                var_adj_x1 = var_adj_x1.unsqueeze(0)
            if var_adj_x2.dim() == 0:
                var_adj_x2 = var_adj_x2.unsqueeze(0)
            
            # Reshape if needed
            if var_adj_x1.shape[0] != x1.shape[0]:
                var_adj_x1 = var_adj_x1.expand(x1.shape[0])
            if var_adj_x2.shape[0] != x2.shape[0]:
                var_adj_x2 = var_adj_x2.expand(x2.shape[0])
            
            # Ensure non-negative values for square root
            var_adj_x1 = torch.clamp(var_adj_x1, min=self.eps)
            var_adj_x2 = torch.clamp(var_adj_x2, min=self.eps)
            
            # Compute causal adjustment term
            if diag:
                # For diagonal, element-wise multiplication
                causal_term = torch.sqrt(var_adj_x1) * torch.sqrt(var_adj_x2)
            else:
                # For full matrix, outer product
                causal_term = torch.outer(torch.sqrt(var_adj_x1), torch.sqrt(var_adj_x2))
            
        except Exception as e:
            # Fallback to zero adjustment if variance function fails
            if diag:
                causal_term = torch.zeros_like(rbf_kernel)
            else:
                causal_term = torch.zeros_like(rbf_kernel)
        
        return rbf_kernel + causal_term


class StandardRBFKernel(RBFKernel):
    """
    Standard RBF kernel for non-causal baseline.
    
    This is a simple wrapper around GPyTorch's RBFKernel for consistency
    with the causal implementation.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)