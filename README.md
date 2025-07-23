# Causal Bayesian Optimization (CBO) - BoTorch Implementation

## 🎯 Project Overview

This project implements Causal Bayesian Optimization (CBO) algorithms using BoTorch, providing comprehensive comparison with standard Bayesian Optimization (BO) across multiple benchmark problems. 

**Project Origin**: Adapted from [VirgiAgl/CausalBayesianOptimization](https://github.com/VirgiAgl/CausalBayesianOptimization), migrated from GPy to BoTorch for better performance and modern PyTorch integration.


## 🚀 Quick Start

### Environment Setup

```bash
# Activate recommended environment (tested)
conda activate BT311  # or your PyTorch 2.0+ environment

# Verify dependencies
python -c "import torch, botorch, gpytorch; print('✅ All dependencies available')"
```

**Required Dependencies**:
- Python 3.11+
- PyTorch 2.5+
- BoTorch 0.10+
- GPyTorch 1.11+
- NumPy, Pandas, Matplotlib, Seaborn, Jupyter

### Quick Testing (10 iterations)

```bash
# Fast test on ToyGraph
./test.sh

# Or run individually
python BO_botorch.py --graph_type ToyGraph --num_trials 10 --seed 0
python CBO_botorch.py --graph_type ToyGraph --num_trials 10 --seed 0
```

### Full Experiments (Recommended)

```bash
# Run complete CBO vs BO comparison (50 iterations, 5 seeds, 4 benchmarks)
python run_experiments.py

# View results
cat results/final_results.json

# Interactive analysis
jupyter notebook CBO_vs_BO_Analysis.ipynb
```


## 📈 Usage Examples

### Individual Algorithm Runs

```bash
# Standard BO on different benchmarks
python BO_botorch.py --graph_type CompleteGraph --num_trials 50 --seed 42
python BO_botorch.py --graph_type CoralGraph --num_trials 100 --seed 0

# Causal BO with different parameters  
python CBO_botorch.py --graph_type SimplifiedCoralGraph --num_trials 50 --seed 1
python CBO_botorch.py --graph_type ToyGraph --num_trials 25 --seed 2
```

### Batch Experiments

```bash
# Multiple seeds for robustness
for seed in {0..4}; do
    python BO_botorch.py --graph_type CoralGraph --num_trials 50 --seed $seed
    python CBO_botorch.py --graph_type CoralGraph --num_trials 50 --seed $seed
done
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

## 📁 Project Structure

```
CausalBayesianOptimization_BoTorch/
├── 🎯 Core Algorithms
│   ├── BO_botorch.py              # Standard BO implementation  
│   ├── CBO_botorch.py             # Causal BO implementation
│   └── run_experiments.py         # Automated experiment runner
│
├── 🧪 Testing & Analysis  
│   ├── test.sh                    # Quick test script
│   └── CBO_vs_BO_Analysis.ipynb   # Comprehensive analysis notebook
│
├── 📊 Data & Results
│   ├── Data/                      # Benchmark datasets
│   │   ├── ToyGraph/             # Simple 3-node graph
│   │   ├── CompleteGraph/        # Complete connectivity
│   │   ├── CoralGraph/           # Real-world marine ecosystem  
│   │   └── SimplifiedCoralGraph/ # Simplified marine model
│   └── results/                  # Experimental results
│       └── final_results.json    # Latest experiment summary
│
├── 🔧 Utilities
│   ├── graphs/                   # Causal graph definitions
│   └── utils_functions/          # BoTorch integration utilities
│       ├── BO_functions_botorch.py
│       ├── causal_acquisition_functions_botorch.py
│       └── causal_kernels_botorch.py
│
└── 📚 Documentation
    ├── README.md                 # This file
    └── CBO_vs_BO_Analysis.ipynb  # Interactive analysis
```
