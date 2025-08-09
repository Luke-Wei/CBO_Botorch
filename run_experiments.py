#!/usr/bin/env python3
"""
Unified Experiment Runner - Complete self-contained CBO vs BO experiments
Includes both serial and parallel execution modes without external dependencies
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
from typing import Optional, List

# Try to import pynvml for better GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

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

def get_gpu_memory_usage_pynvml():
    """Get GPU memory usage using pynvml (recommended method)"""
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
        print(f"pynvml monitoring failed: {e}")
        return None

def monitor_system_resources():
    """Monitor system resources"""
    def _monitor():
        while True:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            
            if torch.cuda.is_available():
                # Priority to pynvml method
                gpu_info_str = get_gpu_memory_usage_pynvml()
                
                if gpu_info_str is None:
                    # Fallback: use nvidia-smi
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
                        except Exception:
                            # Final fallback: torch method (process-level only)
                            memory_used = torch.cuda.memory_allocated(i) / 1024**3
                            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                            gpu_info_str.append(f"GPU{i}: {memory_used:.1f}/{memory_total:.1f}GB (process-level)")
                
                gpu_status = ", ".join(gpu_info_str)
            else:
                gpu_status = "No GPU"
            
            print(f"💻 System status - CPU: {cpu_usage:.1f}%, RAM: {memory_info.percent:.1f}%, {gpu_status}")
            time.sleep(30)  # Update every 30 seconds
    
    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()

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
            
            # Calculate cumulative regrets from instantaneous regrets if not already provided
            if intermediate_data['instantaneous_regrets'] and not intermediate_data['cumulative_regrets']:
                cumulative = 0
                for inst_regret in intermediate_data['instantaneous_regrets']:
                    cumulative += inst_regret
                    intermediate_data['cumulative_regrets'].append(cumulative)
            
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

def check_parallel_capability() -> dict:
    """Check system capability for parallel execution"""
    capability = {
        'can_parallel': True,
        'reasons': [],
        'max_workers': 1,
        'gpu_count': 0,
        'memory_gb': 0
    }
    
    try:
        # Check CPU count
        cpu_count = mp.cpu_count()
        if cpu_count < 2:
            capability['can_parallel'] = False
            capability['reasons'].append("Less than 2 CPU cores available")
        
        # Check memory
        memory_info = psutil.virtual_memory()
        memory_gb = memory_info.total / (1024**3)
        capability['memory_gb'] = memory_gb
        
        if memory_gb < 8:
            capability['can_parallel'] = False
            capability['reasons'].append(f"Insufficient memory: {memory_gb:.1f}GB (need ≥8GB)")
        
        # Check GPU
        if torch.cuda.is_available():
            capability['gpu_count'] = torch.cuda.device_count()
            capability['max_workers'] = min(cpu_count, capability['gpu_count'] * 3)
        else:
            capability['max_workers'] = min(4, cpu_count)
        
        # Check dependencies
        try:
            from concurrent.futures import ProcessPoolExecutor
            import threading
            from queue import Queue
        except ImportError as e:
            capability['can_parallel'] = False
            capability['reasons'].append(f"Missing parallel dependencies: {e}")
            
    except Exception as e:
        capability['can_parallel'] = False
        capability['reasons'].append(f"System check failed: {e}")
    
    return capability

def run_parallel_experiments_internal(algorithms, graph_types, seeds, num_trials):
    """Internal parallel experiment execution"""
    print("🚀 Running parallel execution...")
    
    # Get GPU information
    gpu_info = get_gpu_info()
    
    # Parallel configuration
    max_parallel_per_gpu = 3 if gpu_info["num_gpus"] > 0 else 1
    max_workers = max(1, gpu_info["num_gpus"] * max_parallel_per_gpu) if gpu_info["num_gpus"] > 0 else min(4, mp.cpu_count())
    
    # Create experiment queue
    experiments = create_experiment_queue(algorithms, graph_types, seeds, num_trials)
    experiment_configs = assign_gpu_resources(experiments, gpu_info, max_parallel_per_gpu)
    
    print(f"⚡ Starting parallel execution of {len(experiment_configs)} experiments...")
    print(f"   Max parallelism: {max_workers}")
    
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
    
    return results, time.time() - start_time

def run_serial_experiments_internal(algorithms, graph_types, seeds, num_trials):
    """Internal serial experiment execution"""
    print("🔄 Running serial execution...")
    
    def run_experiment(algorithm, graph_type, num_trials, seed):
        """Run a single experiment"""
        script_name = f"{algorithm}_botorch.py"
        cmd = [
            "python", script_name,
            "--graph_type", graph_type,
            "--num_trials", str(num_trials),
            "--seed", str(seed)
        ]
        
        print(f"Running: {algorithm} on {graph_type} (seed {seed})")
        
        try:
            env = os.environ.copy()
            env['MKL_THREADING_LAYER'] = 'GNU'
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
            
            if result.returncode == 0:
                # Parse results
                output_lines = result.stdout.strip().split('\n')
                final_value = None
                runtime = None
                
                # Parse intermediate process data
                intermediate_data = {
                    'instantaneous_regrets': [],
                    'simple_regrets': [],
                    'cumulative_regrets': []
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
                
                # Calculate cumulative regrets
                if intermediate_data['instantaneous_regrets']:
                    cumulative = 0
                    for inst_regret in intermediate_data['instantaneous_regrets']:
                        cumulative += inst_regret
                        intermediate_data['cumulative_regrets'].append(cumulative)
                
                return {
                    "algorithm": algorithm,
                    "graph_type": graph_type,
                    "seed": seed,
                    "success": True,
                    "final_value": final_value,
                    "runtime": runtime,
                    "intermediate_data": intermediate_data
                }
            else:
                return {
                    "algorithm": algorithm,
                    "graph_type": graph_type,
                    "seed": seed,
                    "success": False, 
                    "error": result.stderr
                }
                
        except Exception as e:
            return {
                "algorithm": algorithm,
                "graph_type": graph_type,
                "seed": seed,
                "success": False, 
                "error": str(e)
            }
    
    all_results = []
    start_time = time.time()
    total_experiments = len(algorithms) * len(graph_types) * len(seeds)
    completed = 0
    
    for algorithm in algorithms:
        for graph_type in graph_types:
            for seed in seeds:
                result = run_experiment(algorithm, graph_type, num_trials, seed)
                all_results.append(result)
                completed += 1
                
                if result['success']:
                    print(f"  ✓ {algorithm}-{graph_type}-{seed}: {result['final_value']:.6f}")
                else:
                    print(f"  ✗ {algorithm}-{graph_type}-{seed}: Failed")
                
                # Progress update
                progress = (completed / total_experiments) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (total_experiments - completed) if completed > 0 else 0
                print(f"📊 Progress: {completed}/{total_experiments} ({progress:.1f}%) - ETA: {eta/60:.1f} minutes")
    
    return all_results, time.time() - start_time

def analyze_and_save_results(results, algorithms, graph_types, seeds, num_trials, total_time, execution_mode):
    """Analyze and save experimental results"""
    print("\n" + "=" * 60)
    print("📈 Experimental Results Analysis")
    print("=" * 60)
    
    # Organize results
    organized_results = {}
    success_count = sum(1 for r in results if r.get('success', False))
    failure_count = len(results) - success_count
    
    print(f"✅ Success: {success_count}, ❌ Failed: {failure_count}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"🚀 Execution mode: {execution_mode}")
    
    for result in results:
        algorithm = result['algorithm']
        graph_type = result['graph_type']
        seed = result['seed']
        
        if algorithm not in organized_results:
            organized_results[algorithm] = {}
        if graph_type not in organized_results[algorithm]:
            organized_results[algorithm][graph_type] = {}
        
        organized_results[algorithm][graph_type][seed] = result
    
    # Statistical analysis
    summary = {}
    for graph_type in graph_types:
        print(f"\n📊 {graph_type}:")
        graph_summary = {}
        
        for algorithm in algorithms:
            values = []
            if algorithm in organized_results and graph_type in organized_results[algorithm]:
                for seed in seeds:
                    if seed in organized_results[algorithm][graph_type]:
                        result = organized_results[algorithm][graph_type][seed]
                        if result.get('success') and result.get('final_value') is not None:
                            values.append(result['final_value'])
            
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"   {algorithm}: {mean_val:.6f} ± {std_val:.6f}")
                graph_summary[algorithm] = {"mean": mean_val, "std": std_val, "values": values}
        
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
            'execution_mode': execution_mode,
            'total_time': total_time,
            'success_rate': success_count / len(results) if results else 0
        },
        'raw_results': organized_results,
        'detailed_results': results,
        'summary': summary
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f'results/Results_{num_trials}_{timestamp}.json'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Experiments completed!")
    print(f"📁 Results saved to: {results_file}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"✅ Success rate: {success_count/len(results)*100:.1f}%")

def main(
    algorithms: Optional[List[str]] = None,
    graph_types: Optional[List[str]] = None,
    num_seeds: int = 5,
    num_trials: int = 100,
    execution_mode: str = 'parallel'
):
    """
    Experiment runner with both serial and parallel execution
    
    Args:
        algorithms: List of algorithms to test ['BO', 'CBO']
        graph_types: List of graph types to test
        num_seeds: Number of random seeds to use (0, 1, 2, ..., num_seeds-1)
        num_trials: Number of optimization trials per experiment
        execution_mode: 'parallel', 'serial', or 'auto'
    """
    
    print("🔧 CBO vs BO Experiment Runner")
    print("=" * 50)
    
    # Set defaults
    algorithms = algorithms or ['BO', 'CBO']
    graph_types = graph_types or ['ToyGraph', 'CompleteGraph', 'CoralGraph', 'SimplifiedCoralGraph'] 
    seeds = list(range(num_seeds))
    
    total_experiments = len(algorithms) * len(graph_types) * len(seeds)
    
    print(f"📋 Experiment Configuration:")
    print(f"   Algorithms: {algorithms}")
    print(f"   Graph types: {graph_types}")
    print(f"   Seeds: {len(seeds)} seeds ({min(seeds)}-{max(seeds)})")
    print(f"   Trials per experiment: {num_trials}")
    print(f"   Total experiments: {total_experiments}")
    print(f"   Execution mode: {execution_mode}")
    
    # Check system capability
    capability = check_parallel_capability()
    
    print(f"\n💻 System Analysis:")
    print(f"   CPU cores: {mp.cpu_count()}")
    print(f"   Memory: {capability['memory_gb']:.1f} GB")
    print(f"   GPUs: {capability['gpu_count']}")
    print(f"   Parallel capable: {'Yes' if capability['can_parallel'] else 'No'}")
    
    if not capability['can_parallel']:
        print(f"   Reasons: {', '.join(capability['reasons'])}")
    
    # Decide execution method
    use_parallel = False
    
    if execution_mode == 'parallel':
        if capability['can_parallel']:
            use_parallel = True
            estimated_speedup = min(capability['max_workers'], total_experiments / 2)
            print(f"\n⚡ Using parallel execution")
            print(f"   Estimated speedup: ~{estimated_speedup:.1f}x")
            print(f"   Max workers: {capability['max_workers']}")
        else:
            use_parallel = False
            print(f"\n🔄 Parallel not available, falling back to serial execution")
            print(f"   Reasons: {', '.join(capability['reasons'])}")
    elif execution_mode == 'serial':
        use_parallel = False
        print(f"\n🔄 Using serial execution")
    elif execution_mode == 'auto':
        # Intelligent decision
        if capability['can_parallel'] and total_experiments >= 4:
            use_parallel = True
            estimated_speedup = min(capability['max_workers'], total_experiments / 2)
            print(f"\n⚡ Auto-selected parallel execution")
            print(f"   Estimated speedup: ~{estimated_speedup:.1f}x")
            print(f"   Max workers: {capability['max_workers']}")
        else:
            use_parallel = False
            if total_experiments < 4:
                print(f"\n🔄 Auto-selected serial execution (few experiments)")
            else:
                print(f"\n🔄 Auto-selected serial execution (system limitations)")
    else:
        raise ValueError(f"Unknown execution_mode: {execution_mode}. Use 'parallel', 'serial', or 'auto'.")
    
    print("=" * 50)
    
    # Execute experiments
    try:
        if use_parallel:
            try:
                results, total_time = run_parallel_experiments_internal(algorithms, graph_types, seeds, num_trials)
                actual_execution_mode = 'parallel'
            except Exception as e:
                print(f"\n❌ Parallel execution failed: {e}")
                if execution_mode == 'parallel':
                    print("\n🔄 Falling back to serial execution...")
                    results, total_time = run_serial_experiments_internal(algorithms, graph_types, seeds, num_trials)
                    actual_execution_mode = 'serial_fallback'
                else:
                    raise
        else:
            results, total_time = run_serial_experiments_internal(algorithms, graph_types, seeds, num_trials)
            actual_execution_mode = 'serial'
        
        # Analyze and save results
        analyze_and_save_results(results, algorithms, graph_types, seeds, num_trials, total_time, actual_execution_mode)
            
    except KeyboardInterrupt:
        print("\n⏹️  Experiments interrupted by user")
        return
    except Exception as e:
        print(f"\n💥 Experiment runner failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified CBO vs BO experiments')
    parser.add_argument('--algorithms', nargs='+', default=['BO', 'CBO'], 
                       help='Algorithms to test (default: %(default)s)')
    parser.add_argument('--graph_types', nargs='+', 
                       default=['ToyGraph', 'CompleteGraph', 'CoralGraph', 'SimplifiedCoralGraph'],
                       help='Graph types to test (default: %(default)s)')
    parser.add_argument('--num_seeds', type=int, default=50,
                       help='Number of random seeds to use (0, 1, ..., num_seeds-1) (default: %(default)s)')
    parser.add_argument('--num_trials', type=int, default=100,
                       help='Number of trials per experiment (default: %(default)s)')
    parser.add_argument('--execution_mode', choices=['parallel', 'serial', 'auto'], 
                       default='parallel',
                       help='Execution mode: parallel (default), serial, or auto-detect')
    
    args = parser.parse_args()
    
    main(
        algorithms=args.algorithms,
        graph_types=args.graph_types, 
        num_seeds=args.num_seeds,
        num_trials=args.num_trials,
        execution_mode=args.execution_mode
    )