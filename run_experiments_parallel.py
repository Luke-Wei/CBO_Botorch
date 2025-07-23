#!/usr/bin/env python3
"""
CausalBayesianOptimization - Parallel Experiment Script
Run CBO vs BO complete comparison experiments with multi-GPU parallel acceleration
"""

import subprocess
import json
import time
import os
import numpy as np
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from queue import Queue
import psutil

# 尝试导入pynvml进行更专业的GPU监控
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("💡 提示: 安装 pynvml 库可获得更精确的GPU监控: pip install pynvml")

def get_gpu_memory_usage_pynvml():
    """使用pynvml获取GPU内存使用情况（推荐方法）"""
    if not PYNVML_AVAILABLE:
        return None
    
    try:
        pynvml.nvmlInit()
        gpu_info = []
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            memory_used_gb = memory_info.used / 1024**3
            memory_total_gb = memory_info.total / 1024**3
            gpu_util = utilization.gpu
            
            gpu_info.append(f"GPU{i}: {memory_used_gb:.1f}/{memory_total_gb:.1f}GB ({gpu_util}%)")
        
        return gpu_info
    except Exception as e:
        print(f"pynvml监控失败: {e}")
        return None

def get_gpu_info():
    """Get GPU information"""
    if not torch.cuda.is_available():
        return {"num_gpus": 0, "gpu_names": [], "gpu_memory": []}
    
    num_gpus = torch.cuda.device_count()
    gpu_names = []
    gpu_memory = []
    
    for i in range(num_gpus):
        gpu_names.append(torch.cuda.get_device_name(i))
        gpu_memory.append(torch.cuda.get_device_properties(i).total_memory / 1024**3)
    
    return {
        "num_gpus": num_gpus,
        "gpu_names": gpu_names,
        "gpu_memory": gpu_memory
    }

def run_single_experiment(experiment_config):
    """Run single experiment worker process function"""
    algorithm, graph_type, num_trials, seed, gpu_id = experiment_config
    
    script_name = f"{algorithm}_botorch.py"
    cmd = [
        "python", script_name,
        "--graph_type", graph_type,
        "--num_trials", str(num_trials),
        "--seed", str(seed)
    ]
    
    experiment_id = f"{algorithm}-{graph_type}-{seed}"
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['MKL_THREADING_LAYER'] = 'GNU'
        if gpu_id is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        print(f"Starting experiment {experiment_id} (GPU {gpu_id if gpu_id is not None else 'CPU'})")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            # Parse results
            output_lines = result.stdout.strip().split('\n')
            final_value = None
            runtime = None
            
            # Parse intermediate process data
            intermediate_data = {
                'instantaneous_regrets': [],
                'simple_regrets': [],
                'cumulative_regrets': [],
                'x_gaps': [],
                'y_gaps': []
            }
            
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
                elif "Instantaneous regret:" in line:
                    try:
                        val = float(line.split(":")[-1].strip())
                        intermediate_data['instantaneous_regrets'].append(val)
                    except:
                        pass
                elif "Simple regret:" in line:
                    try:
                        val = float(line.split(":")[-1].strip())
                        intermediate_data['simple_regrets'].append(val)
                    except:
                        pass
                elif "Cumulative regret:" in line:
                    try:
                        val = float(line.split(":")[-1].strip())
                        intermediate_data['cumulative_regrets'].append(val)
                    except:
                        pass
            
            print(f"✅ Completed experiment {experiment_id}: {final_value:.6f} ({elapsed_time:.1f}s) - {len(intermediate_data['simple_regrets'])} regret points")
            
            return {
                "experiment_id": experiment_id,
                "algorithm": algorithm,
                "graph_type": graph_type,
                "seed": seed,
                "gpu_id": gpu_id,
                "success": True,
                "final_value": final_value,
                "runtime": runtime,
                "wall_time": elapsed_time,
                "intermediate_data": intermediate_data
            }
        else:
            print(f"❌ Experiment failed {experiment_id}: {result.stderr[:100]}")
            return {
                "experiment_id": experiment_id,
                "algorithm": algorithm,
                "graph_type": graph_type,
                "seed": seed,
                "gpu_id": gpu_id,
                "success": False,
                "error": result.stderr,
                "wall_time": elapsed_time
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment timeout {experiment_id}")
        return {
            "experiment_id": experiment_id,
            "algorithm": algorithm,
            "graph_type": graph_type,
            "seed": seed,
            "gpu_id": gpu_id,
            "success": False,
            "error": "Timeout",
            "wall_time": elapsed_time if 'elapsed_time' in locals() else 3600
        }
    except Exception as e:
        print(f"💥 Experiment exception {experiment_id}: {str(e)}")
        return {
            "experiment_id": experiment_id,
            "algorithm": algorithm,
            "graph_type": graph_type,
            "seed": seed,
            "gpu_id": gpu_id,
            "success": False,
            "error": str(e),
            "wall_time": elapsed_time if 'elapsed_time' in locals() else 0
        }

def create_experiment_queue(algorithms, graph_types, seeds, num_trials):
    """Create experiment queue"""
    experiments = []
    for algorithm in algorithms:
        for graph_type in graph_types:
            for seed in seeds:
                experiments.append((algorithm, graph_type, num_trials, seed))
    return experiments

def assign_gpu_resources(experiments, gpu_info, max_parallel_per_gpu=4):
    """Assign GPU resources for experiments"""
    if gpu_info["num_gpus"] == 0:
        # No GPU available, use CPU
        return [(exp[0], exp[1], exp[2], exp[3], None) for exp in experiments]
    
    experiment_configs = []
    gpu_queue = []
    
    # Create multiple slots for each GPU
    for gpu_id in range(gpu_info["num_gpus"]):
        for _ in range(max_parallel_per_gpu):
            gpu_queue.append(gpu_id)
    
    # Cyclically assign GPUs
    for i, exp in enumerate(experiments):
        gpu_id = gpu_queue[i % len(gpu_queue)]
        experiment_configs.append((exp[0], exp[1], exp[2], exp[3], gpu_id))
    
    return experiment_configs

def monitor_system_resources():
    """Monitor system resources"""
    def _monitor():
        while True:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            
            if torch.cuda.is_available():
                # 优先使用pynvml方法
                gpu_info_str = get_gpu_memory_usage_pynvml()
                
                if gpu_info_str is None:
                    # 备选方案1: 使用nvidia-smi
                    gpu_info_str = []
                    for i in range(torch.cuda.device_count()):
                        try:
                            result = subprocess.run([
                                'nvidia-smi', 
                                '--id=' + str(i),
                                '--query-gpu=memory.used,memory.total,utilization.gpu',
                                '--format=csv,noheader,nounits'
                            ], capture_output=True, text=True, timeout=5)
                            
                            if result.returncode == 0:
                                memory_used, memory_total, gpu_util = result.stdout.strip().split(', ')
                                memory_used_gb = float(memory_used) / 1024
                                memory_total_gb = float(memory_total) / 1024
                                gpu_info_str.append(f"GPU{i}: {memory_used_gb:.1f}/{memory_total_gb:.1f}GB ({gpu_util}%)")
                            else:
                                gpu_info_str.append(f"GPU{i}: N/A")
                        except Exception as e:
                            # 备选方案2: 回退到torch方法（仅显示当前进程）
                            memory_used = torch.cuda.memory_allocated(i) / 1024**3
                            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                            gpu_info_str.append(f"GPU{i}: {memory_used:.1f}/{memory_total:.1f}GB (进程级)")
                
                gpu_status = ", ".join(gpu_info_str)
            else:
                gpu_status = "No GPU"
            
            print(f"💻 System status - CPU: {cpu_usage:.1f}%, RAM: {memory_info.percent:.1f}%, {gpu_status}")
            time.sleep(30)  # Update every 30 seconds
    
    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()

def main():
    """Main function"""
    print("🚀 Starting CausalBayesianOptimization parallel experiments")
    print("=" * 60)
    
    # Get GPU information
    gpu_info = get_gpu_info()
    print(f"🔧 GPU information:")
    if gpu_info["num_gpus"] > 0:
        for i, (name, memory) in enumerate(zip(gpu_info["gpu_names"], gpu_info["gpu_memory"])):
            print(f"   GPU {i}: {name} ({memory:.1f} GB)")
    else:
        print("   No available GPU, will use CPU")
    
    # Experiment configuration
    algorithms = ['BO', 'CBO']
    graph_types = ['ToyGraph', 'CompleteGraph', 'CoralGraph', 'SimplifiedCoralGraph']
    seeds = list(range(50))  
    num_trials = 100  # Quick test with fewer trials
    
    # Parallel configuration
    max_parallel_per_gpu = 3 if gpu_info["num_gpus"] > 0 else 1  # Max 3 parallel experiments per GPU
    max_workers = max(1, gpu_info["num_gpus"] * max_parallel_per_gpu) if gpu_info["num_gpus"] > 0 else min(4, mp.cpu_count())
    
    print(f"📋 Experiment configuration:")
    print(f"   Algorithms: {algorithms}")
    print(f"   Graph types: {graph_types}")
    print(f"   Seeds: {len(seeds)} seeds ({min(seeds)}-{max(seeds)})")
    print(f"   Trials per experiment: {num_trials}")
    print(f"   Total experiments: {len(algorithms) * len(graph_types) * len(seeds)}")
    print(f"   Max parallelism: {max_workers}")
    print(f"   Parallel per GPU: {max_parallel_per_gpu}")
    
    # Create experiment queue
    experiments = create_experiment_queue(algorithms, graph_types, seeds, num_trials)
    experiment_configs = assign_gpu_resources(experiments, gpu_info, max_parallel_per_gpu)
    
    print(f"\n⚡ Starting parallel execution of {len(experiment_configs)} experiments...")
    print("=" * 60)
    
    # Start system resource monitoring
    monitor_system_resources()
    
    # Execute parallel experiments
    results = []
    completed_count = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_config = {executor.submit(run_single_experiment, config): config 
                          for config in experiment_configs}
        
        # Process completed tasks
        for future in as_completed(future_to_config):
            result = future.result()
            results.append(result)
            completed_count += 1
            
            elapsed_time = time.time() - start_time
            progress = (completed_count / len(experiment_configs)) * 100
            eta = (elapsed_time / completed_count) * (len(experiment_configs) - completed_count)
            
            print(f"📊 Progress: {completed_count}/{len(experiment_configs)} ({progress:.1f}%) - ETA: {eta/60:.1f} minutes")
    
    total_time = time.time() - start_time
    
    # Organize results
    print("\n" + "=" * 60)
    print("📈 Experimental Results Analysis")
    print("=" * 60)
    
    # Organize results in original format
    organized_results = {}
    for result in results:
        algorithm = result['algorithm']
        graph_type = result['graph_type']
        seed = result['seed']
        
        if algorithm not in organized_results:
            organized_results[algorithm] = {}
        if graph_type not in organized_results[algorithm]:
            organized_results[algorithm][graph_type] = {}
        
        organized_results[algorithm][graph_type][seed] = {
            'success': result['success'],
            'final_value': result.get('final_value'),
            'runtime': result.get('runtime'),
            'wall_time': result.get('wall_time'),
            'gpu_id': result.get('gpu_id')
        }
    
    # Statistical analysis
    summary = {}
    success_count = sum(1 for r in results if r['success'])
    failure_count = len(results) - success_count
    
    print(f"✅ Success: {success_count}, ❌ Failed: {failure_count}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"🚀 Speedup: ~{len(experiment_configs) * 5 / (total_time/60):.1f}x (estimated)")
    
    for graph_type in graph_types:
        print(f"\n📊 {graph_type}:")
        graph_summary = {}
        
        for algorithm in algorithms:
            values = []
            wall_times = []
            
            if algorithm in organized_results and graph_type in organized_results[algorithm]:
                for seed in seeds:
                    if seed in organized_results[algorithm][graph_type]:
                        result = organized_results[algorithm][graph_type][seed]
                        if result['success'] and result['final_value'] is not None:
                            values.append(result['final_value'])
                        if result.get('wall_time'):
                            wall_times.append(result['wall_time'])
            
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                mean_time = np.mean(wall_times) if wall_times else 0
                print(f"   {algorithm}: {mean_val:.6f} ± {std_val:.6f} (avg {mean_time:.1f}s)")
                graph_summary[algorithm] = {
                    "mean": mean_val, 
                    "std": std_val, 
                    "values": values,
                    "mean_wall_time": mean_time
                }
        
        # Calculate improvement
        if 'BO' in graph_summary and 'CBO' in graph_summary:
            bo_mean = graph_summary['BO']['mean']
            cbo_mean = graph_summary['CBO']['mean']
            improvement = ((cbo_mean - bo_mean) / abs(bo_mean)) * 100
            
            if improvement > 0:
                print(f"   → CBO improvement: {improvement:.1f}%")
            else:
                print(f"   → BO advantage: {-improvement:.1f}%")
            
            graph_summary['improvement'] = improvement
        
        summary[graph_type] = graph_summary
    
    # Save results
    os.makedirs('results', exist_ok=True)
    final_results = {
        'experiment_config': {
            'algorithms': algorithms,
            'graph_types': graph_types,
            'seeds': seeds,
            'num_trials': num_trials,
            'parallel_config': {
                'max_workers': max_workers,
                'max_parallel_per_gpu': max_parallel_per_gpu,
                'gpu_info': gpu_info
            },
            'total_time': total_time,
            'success_rate': success_count / len(results)
        },
        'raw_results': organized_results,
        'detailed_results': results,
        'summary': summary
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f'results/parallel_results_100_{timestamp}.json'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Parallel experiments completed!")
    print(f"📁 Results saved to: {results_file}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"✅ Success rate: {success_count/len(results)*100:.1f}%")

if __name__ == "__main__":
    main()