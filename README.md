# Causal Bayesian Optimization (CBO) - BoTorch Implementation

## 🎯 Project Overview

This project implements Causal Bayesian Optimization (CBO) algorithms using BoTorch, providing fair comparison with standard Bayesian Optimization (BO). 

**Project Origin**: This project is adapted from [VirgiAgl/CausalBayesianOptimization](https://github.com/VirgiAgl/CausalBayesianOptimization), migrating from the original GPy-based implementation to a BoTorch-based implementation for better performance and modern deep learning framework support.

## 🚀 Quick Start

### Environment Setup

```bash
# Activate PyTorch environment
conda activate your_pytorch_env  # or your PyTorch environment

# Install dependencies
pip install torch botorch gpytorch pandas numpy matplotlib seaborn jupyter
```

### Quick Testing

```bash
# Run quick test
./test.sh

# Or manual testing
python BO_botorch.py --graph_type ToyGraph --num_trials 10 --seed 0
python CBO_botorch.py --graph_type ToyGraph --num_trials 10 --seed 0
```

### Full Experiments

```bash
# Run complete CBO vs BO comparison experiments
python run_experiments.py

# View results
cat results/final_results.json
```

## 📈 Usage & Analysis

### Command Line Results

```bash
# View latest results
cat results/final_results.json

# Extract key data
python -c "import json; data=json.load(open('results/final_results.json')); print(data['summary'])"
```

### Custom Experiments

```bash
# Run specific algorithms
python BO_botorch.py --graph_type CompleteGraph --num_trials 100 --seed 42
python CBO_botorch.py --graph_type CoralGraph --num_trials 50 --seed 0

# GPU acceleration (if available)
python CBO_botorch.py --graph_type ToyGraph --device cuda --num_trials 100
```

### Parameter Configuration

- `--graph_type`: Choose from ToyGraph, CompleteGraph, CoralGraph, SimplifiedCoralGraph
- `--num_trials`: Number of optimization iterations (default: 100)
- `--seed`: Random seed for reproducibility (default: 0)
- `--device`: PyTorch device (auto, cpu, cuda)

## 📋 Dependencies

- Python 3.8+
- PyTorch 1.13+
- BoTorch 0.8+
- GPyTorch 1.9+
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn (for visualization)
- Jupyter (for interactive analysis)

## 📄 License

This project follows the same license terms as the original implementation. See LICENSE file for details.

---

**🎯 Fair comparison implementation with complete consistency to original GPy benchmarks**

