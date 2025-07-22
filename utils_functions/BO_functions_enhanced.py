## Enhanced GPy implementation with improved preprocessing and optimization
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools
from numpy.random import randn
import copy
import seaborn as sns

import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper

from .causal_kernels import CausalRBF


class EnhancedDataPreprocessor:
    """
    Enhanced data preprocessing to match BoTorch capabilities.
    """
    
    def __init__(self, normalize_inputs=True, standardize_outputs=True):
        self.normalize_inputs = normalize_inputs
        self.standardize_outputs = standardize_outputs
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        self.fitted = False
    
    def fit(self, X, Y):
        """Fit preprocessing parameters."""
        if self.normalize_inputs:
            self.input_mean = np.mean(X, axis=0, keepdims=True)
            self.input_std = np.std(X, axis=0, keepdims=True)
            # Avoid division by zero
            self.input_std = np.where(self.input_std < 1e-8, 1.0, self.input_std)
        
        if self.standardize_outputs:
            self.output_mean = np.mean(Y, axis=0, keepdims=True)
            self.output_std = np.std(Y, axis=0, keepdims=True)
            # Avoid division by zero
            self.output_std = np.where(self.output_std < 1e-8, 1.0, self.output_std)
        
        self.fitted = True
    
    def transform_inputs(self, X):
        """Transform inputs."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        if self.normalize_inputs:
            return (X - self.input_mean) / self.input_std
        return X
    
    def transform_outputs(self, Y):
        """Transform outputs."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        if self.standardize_outputs:
            return (Y - self.output_mean) / self.output_std
        return Y
    
    def inverse_transform_outputs(self, Y_transformed):
        """Inverse transform outputs."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before inverse transform")
        
        if self.standardize_outputs:
            return Y_transformed * self.output_std + self.output_mean
        return Y_transformed
    
    def inverse_transform_variance(self, var_transformed):
        """Inverse transform variance."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before inverse transform")
        
        if self.standardize_outputs:
            return var_transformed * (self.output_std ** 2)
        return var_transformed


class EnhancedGPyModel:
    """
    Enhanced GPy model wrapper with better optimization and error handling.
    """
    
    def __init__(self, gpy_model, preprocessor=None):
        self.gpy_model = gpy_model
        self.preprocessor = preprocessor
        self.fitted = False
    
    def optimize(self, num_restarts=5, max_iters=1000, verbose=False):
        """Enhanced optimization with multiple restarts."""
        best_model = None
        best_likelihood = -np.inf
        
        for restart in range(num_restarts):
            try:
                # Randomize hyperparameters for restart
                if restart > 0:
                    self.gpy_model.randomize()
                
                # Optimize
                self.gpy_model.optimize(messages=verbose, max_iters=max_iters)
                
                # Check if this is the best model so far
                current_likelihood = self.gpy_model.log_likelihood()
                if current_likelihood > best_likelihood:
                    best_likelihood = current_likelihood
                    best_model = self.gpy_model.copy()
                
            except Exception as e:
                if verbose:
                    print(f"Optimization restart {restart} failed: {e}")
                continue
        
        # Use best model
        if best_model is not None:
            self.gpy_model = best_model
            self.fitted = True
        
        return self
    
    def predict(self, X_new, full_cov=False):
        """Make predictions with proper preprocessing."""
        if self.preprocessor is not None:
            X_new_transformed = self.preprocessor.transform_inputs(X_new)
        else:
            X_new_transformed = X_new
        
        # Get predictions
        mean, var = self.gpy_model.predict(X_new_transformed, full_cov=full_cov)
        
        # Inverse transform
        if self.preprocessor is not None:
            mean = self.preprocessor.inverse_transform_outputs(mean)
            if full_cov:
                # For full covariance, need to scale properly
                var = self.preprocessor.inverse_transform_variance(var)
            else:
                var = self.preprocessor.inverse_transform_variance(var)
        
        return mean, var
    
    def log_likelihood(self):
        """Get log likelihood."""
        return self.gpy_model.log_likelihood()


def define_initial_data_BO_enhanced(
    interventional_data, 
    num_interventions, 
    intervention_sets, 
    name_index,
    use_preprocessing=True
):
    """
    Enhanced initial data preparation with optional preprocessing.
    """
    data_x = (interventional_data[0][len(intervention_sets)]).copy()
    data_y = (np.asarray(interventional_data[0][len(intervention_sets)+1])).copy()
    all_data = np.concatenate((data_x, data_y), axis=1)

    # Reset the global seed 
    state = np.random.get_state()
    np.random.seed(name_index)
    np.random.shuffle(all_data)
    np.random.set_state(state)

    data_x = all_data[:num_interventions, :len(intervention_sets)]
    data_y = all_data[:num_interventions, len(intervention_sets):]

    # Optional preprocessing
    preprocessor = None
    if use_preprocessing:
        preprocessor = EnhancedDataPreprocessor()
        preprocessor.fit(data_x, data_y)

    min_y = np.min(data_y)
    min_intervention_value = np.transpose(all_data[np.where(data_y == min_y)[0][0]][:len(intervention_sets)][:, np.newaxis])

    return data_x, data_y, min_intervention_value, min_y, preprocessor


def update_BO_models_enhanced(
    mean_function, 
    var_function, 
    data_x, 
    data_y, 
    Causal_prior,
    use_preprocessing=True,
    enhanced_optimization=True,
    num_restarts=5
):    
    """
    Enhanced BO model creation with better optimization and preprocessing.
    """
    # Set up preprocessing
    preprocessor = None
    if use_preprocessing:
        preprocessor = EnhancedDataPreprocessor()
        preprocessor.fit(data_x, data_y)
        data_x_processed = preprocessor.transform_inputs(data_x)
        data_y_processed = preprocessor.transform_outputs(data_y)
    else:
        data_x_processed = data_x
        data_y_processed = data_y
    
    # Create model
    if Causal_prior == False:
        gpy_model = GPy.models.GPRegression(
            data_x_processed, 
            data_y_processed, 
            GPy.kern.RBF(data_x_processed.shape[1], lengthscale=1., variance=1.), 
            noise_var=1e-10
        )
    else:    
        # Enhanced mean function handling
        if mean_function is not None:
            mf = GPy.core.Mapping(data_x_processed.shape[1], 1)
            if use_preprocessing and preprocessor is not None:
                # Adapt mean function for preprocessed data
                def adapted_mean_function(x):
                    # Transform back to original scale for mean function
                    x_orig = preprocessor.input_mean + x * preprocessor.input_std
                    mean_orig = mean_function(x_orig)
                    # Transform to processed scale
                    return preprocessor.transform_outputs(mean_orig)
                mf.f = lambda x: adapted_mean_function(x)
            else:
                mf.f = lambda x: mean_function(x)
            mf.update_gradients = lambda a, b: None
        else:
            mf = None

        # Enhanced variance function handling
        if var_function is not None and use_preprocessing and preprocessor is not None:
            def adapted_var_function(x):
                # Transform back to original scale for variance function
                x_orig = preprocessor.input_mean + x * preprocessor.input_std
                return var_function(x_orig)
            causal_kernel = CausalRBF(
                data_x_processed.shape[1], 
                variance_adjustment=adapted_var_function, 
                lengthscale=1., 
                variance=1., 
                ARD=False
            )
        else:
            causal_kernel = CausalRBF(
                data_x_processed.shape[1], 
                variance_adjustment=var_function, 
                lengthscale=1., 
                variance=1., 
                ARD=False
            )

        gpy_model = GPy.models.GPRegression(
            data_x_processed, 
            data_y_processed, 
            causal_kernel, 
            noise_var=1e-10, 
            mean_function=mf
        )
    
    # Enhanced optimization
    enhanced_model = EnhancedGPyModel(gpy_model, preprocessor)
    if enhanced_optimization:
        enhanced_model.optimize(num_restarts=num_restarts, verbose=False)
    else:
        # Standard optimization
        gpy_model.optimize()
        enhanced_model.fitted = True
    
    # Wrap for emukit compatibility
    model = GPyModelWrapper(enhanced_model.gpy_model)
    model.enhanced_model = enhanced_model  # Keep reference to enhanced functionality
    
    return model


def robust_model_fitting(model, max_attempts=3, verbose=False):
    """
    Robust model fitting with fallback strategies.
    """
    for attempt in range(max_attempts):
        try:
            if hasattr(model, 'enhanced_model'):
                model.enhanced_model.optimize(num_restarts=3, verbose=verbose)
            else:
                model.model.optimize()
            return True
        except Exception as e:
            if verbose:
                print(f"Fitting attempt {attempt + 1} failed: {e}")
            if attempt < max_attempts - 1:
                # Try with different initialization
                if hasattr(model, 'model'):
                    model.model.randomize()
                continue
            else:
                if verbose:
                    print("All fitting attempts failed, using unfitted model")
                return False
    return False


def enhanced_prediction_with_uncertainty(model, X_new, confidence_level=0.95):
    """
    Enhanced prediction with proper uncertainty quantification.
    """
    try:
        if hasattr(model, 'enhanced_model'):
            mean, var = model.enhanced_model.predict(X_new)
        else:
            mean, var = model.predict(X_new)
        
        # Compute confidence intervals
        std = np.sqrt(var)
        z_score = scipy.stats.norm.ppf((1 + confidence_level) / 2)
        
        lower_bound = mean - z_score * std
        upper_bound = mean + z_score * std
        
        return {
            'mean': mean,
            'variance': var,
            'std': std,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level
        }
    except Exception as e:
        print(f"Enhanced prediction failed: {e}")
        # Fallback to basic prediction
        mean, var = model.predict(X_new)
        return {
            'mean': mean,
            'variance': var,
            'std': np.sqrt(var),
            'lower_bound': None,
            'upper_bound': None,
            'confidence_level': None
        }