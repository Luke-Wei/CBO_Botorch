#!/bin/bash
# CausalBayesianOptimization 测试脚本

echo "🧪 开始CausalBayesianOptimization测试"
echo "======================================"

# 设置环境
export MKL_THREADING_LAYER=GNU

echo "📋 测试配置:"
echo "- 图类型: ToyGraph"
echo "- 迭代次数: 10 (快速测试)"
echo "- 随机种子: 0"
echo ""

echo "🔧 测试标准BO..."
python BO_botorch.py --graph_type ToyGraph --num_trials 10 --seed 0

echo ""
echo "🔧 测试因果BO..."
python CBO_botorch.py --graph_type ToyGraph --num_trials 10 --seed 0

echo ""
echo "✅ 测试完成!"
echo ""
echo "💡 运行完整实验: python run_experiments.py"
echo "📊 查看分析: jupyter notebook CBO_vs_BO_Analysis.ipynb"