# Causal Bayesian Optimization (CBO) - BoTorch Implementation

## 🎯 Project Overview

This project implements Causal Bayesian Optimization (CBO) algorithms using BoTorch, providing comprehensive comparison with standard Bayesian Optimization (BO) across multiple benchmark problems. 

**Project Origin**: Adapted from [VirgiAgl/CausalBayesianOptimization](https://github.com/VirgiAgl/CausalBayesianOptimization), migrated from GPy to BoTorch for better performance and modern PyTorch integration.


## 🚀 Quick Start

### Environment Setup

# Activate recommended environment

**Required Dependencies**:
- Python 3.11+
- PyTorch 2.5+
- BoTorch 0.10+
- GPyTorch 1.11+
- NumPy, Pandas, Matplotlib, Seaborn, Jupyter


### Full Experiments

```bash
# Run complete CBO vs BO comparison (50 iterations, 5 seeds, 4 benchmarks)
python run_experiments.py

# View results
cat results/final_results.json

# Interactive analysis
jupyter notebook CBO_vs_BO_Analysis.ipynb
```



## 🔧 Parameter Configuration

**Main Scripts**:
- `BO_botorch.py`: Standard Bayesian Optimization
- `CBO_botorch.py`: Causal Bayesian Optimization  
- `run_experiments.py`: Automated batch experiments

**Key Parameters**:
- `--graph_type`: `ToyGraph`, `CompleteGraph`, `CoralGraph`, `SimplifiedCoralGraph`
- `--num_trials`: Optimization iterations (default: 50, recommended: 50-100)
- `--seed`: Random seed for reproducibility (0-9 recommended)
- `--device`: PyTorch device (`cpu`, `cuda`, `auto`)

**Experiment Configuration** (in `run_experiments.py`):
- Algorithms: `['BO', 'CBO']`
- Seeds: `[0, 1, 2, 3, 4]` (5 replications)
- Iterations: `50` (production setting)
- Benchmarks: All 4 graph types
