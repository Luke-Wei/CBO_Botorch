# Causal Bayesian Optimization (CBO)

## 🎯 项目概述

本项目实现了基于BoTorch的因果贝叶斯优化(CBO)算法，与标准贝叶斯优化(BO)进行公平对比。

**项目来源**: 本项目从 [VirgiAgl/CausalBayesianOptimization](https://github.com/VirgiAgl/CausalBayesianOptimization) 修改而来，将原始基于GPy的实现改为基于BoTorch的实现，提供更好的性能和现代化的深度学习框架支持。


## 🚀 快速开始

### 环境要求

```bash
# 激活PyTorch环境

# 安装依赖包
pip install torch botorch gpytorch pandas numpy matplotlib seaborn jupyter
```

### 快速测试

```bash
# 运行快速测试
./test.sh

# 或手动测试
python BO_botorch.py --graph_type ToyGraph --num_trials 10 --seed 0
python CBO_botorch.py --graph_type ToyGraph --num_trials 10 --seed 0
```

### 完整实验

```bash
# 运行完整的CBO vs BO对比实验
python run_experiments.py

# 查看结果
cat results/final_results.json
```

## 📊 实验结果

To Update ...

## 📁 项目结构

```
CausalBayesianOptimization/
├── 🎯 核心算法
│   ├── BO_botorch.py              # 标准BO实现
│   ├── CBO_botorch.py             # 因果BO实现  
│   └── run_experiments.py         # 主实验脚本
│
├── 🧪 测试和分析
│   ├── test.sh                    # 快速测试脚本
│   └── CBO_vs_BO_Analysis.ipynb   # 交互式分析
│
├── 📊 数据和结果
│   ├── graphs/                    # 因果图定义
│   ├── Data/                      # 实验数据
│   └── results/                   # 实验结果
│
└── 🔧 工具函数
    └── utils_functions/           # BoTorch集成工具
```

## 🔬 算法原理

### 因果贝叶斯优化 (CBO)


## 📈 使用分析

### 命令行结果查看
```bash
# 查看最新结果
cat results/final_results.json

# 提取关键数据
python -c "import json; data=json.load(open('results/final_results.json')); print(data['summary'])"
```

