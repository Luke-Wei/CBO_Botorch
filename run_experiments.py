#!/usr/bin/env python3
"""
CausalBayesianOptimization - 主实验脚本
运行CBO vs BO完整对比实验
"""

import subprocess
import json
import time
import os
import numpy as np
import torch

# Set up device information
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Experiments using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("   Using CPU (no CUDA available)")

def run_experiment(algorithm, graph_type, num_trials=50, seed=0):
    """运行单个实验"""
    script_name = f"{algorithm}_botorch.py"
    cmd = [
        "python", script_name,
        "--graph_type", graph_type,
        "--num_trials", str(num_trials),
        "--seed", str(seed)
    ]
    
    print(f"运行: {algorithm} on {graph_type} (seed {seed})")
    
    try:
        env = os.environ.copy()
        env['MKL_THREADING_LAYER'] = 'GNU'
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
        
        if result.returncode == 0:
            # 解析结果
            output_lines = result.stdout.strip().split('\n')
            final_value = None
            runtime = None
            
            for line in output_lines:
                if "Final best value:" in line:
                    try:
                        final_value = float(line.split(":")[-1].strip())
                    except:
                        pass
                elif "Runtime:" in line:
                    try:
                        runtime = float(line.split(":")[-1].strip().split()[0])
                    except:
                        pass
            
            return {
                "success": True,
                "final_value": final_value,
                "runtime": runtime
            }
        else:
            return {"success": False, "error": result.stderr}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    """主实验函数"""
    print("🚀 开始CausalBayesianOptimization实验")
    print("=" * 50)
    
    algorithms = ['BO', 'CBO']
    graph_types = ['ToyGraph', 'CompleteGraph', 'CoralGraph', 'SimplifiedCoralGraph']
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_trials = 100
    
    print(f"配置: {len(algorithms)} 算法 × {len(graph_types)} 图 × {len(seeds)} 种子 × {num_trials} 迭代")
    print(f"总实验数: {len(algorithms) * len(graph_types) * len(seeds)}")
    
    all_results = {}
    start_time = time.time()
    
    for algorithm in algorithms:
        all_results[algorithm] = {}
        for graph_type in graph_types:
            all_results[algorithm][graph_type] = {}
            for seed in seeds:
                result = run_experiment(algorithm, graph_type, num_trials, seed)
                all_results[algorithm][graph_type][seed] = result
                
                if result['success']:
                    print(f"  ✓ {algorithm}-{graph_type}-{seed}: {result['final_value']:.6f}")
                else:
                    print(f"  ✗ {algorithm}-{graph_type}-{seed}: 失败")
    
    # 分析结果
    print("\n" + "=" * 50)
    print("📊 实验结果分析")
    print("=" * 50)
    
    summary = {}
    for graph_type in graph_types:
        print(f"\n{graph_type}:")
        graph_summary = {}
        
        for algorithm in algorithms:
            values = []
            for seed in seeds:
                result = all_results[algorithm][graph_type][seed]
                if result['success'] and result['final_value'] is not None:
                    values.append(result['final_value'])
            
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {algorithm}: {mean_val:.6f} ± {std_val:.6f}")
                graph_summary[algorithm] = {"mean": mean_val, "std": std_val, "values": values}
        
        # 计算改进
        if 'BO' in graph_summary and 'CBO' in graph_summary:
            bo_mean = graph_summary['BO']['mean']
            cbo_mean = graph_summary['CBO']['mean']
            
            if graph_type == 'ToyGraph':
                improvement = ((bo_mean - cbo_mean) / abs(bo_mean)) * 100
            else:
                improvement = ((bo_mean - cbo_mean) / abs(bo_mean)) * 100
            
            if improvement > 0:
                print(f"  → CBO改进: {improvement:.1f}%")
            else:
                print(f"  → BO优势: {-improvement:.1f}%")
            
            graph_summary['improvement'] = improvement
        
        summary[graph_type] = graph_summary
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    final_results = {
        'experiment_config': {
            'algorithms': algorithms,
            'graph_types': graph_types,
            'seeds': seeds,
            'num_trials': num_trials,
            'total_time': time.time() - start_time
        },
        'raw_results': all_results,
        'summary': summary
    }
    
    with open('results/final_results_100.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n🎉 实验完成! 结果保存至: results/final_results.json")
    print(f"总用时: {time.time() - start_time:.1f} 秒")

if __name__ == "__main__":
    main()