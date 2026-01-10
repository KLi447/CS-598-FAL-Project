import pandas as pd
import numpy as np
import yaml
import os
import pynvml
import subprocess
import selectors
import signal
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

class GPUMonitor(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.records = []
        self.available = False
        
        if pynvml:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
                self.available = True
            except Exception as e:
                print(f"Failed to initialize NVML: {e}")

    def run(self):
        if not self.available:
            return

        while not self.stop_event.is_set():
            timestamp = time.time()
            snapshot = {'timestamp': timestamp, 'gpu_utils': [], 'mem_utils': []}
            
            for i, handle in enumerate(self.handles):
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    snapshot['gpu_utils'].append(util.gpu)  
                    snapshot['mem_utils'].append(util.memory)
                except Exception:
                    snapshot['gpu_utils'].append(0)
                    snapshot['mem_utils'].append(0)
            
            self.records.append(snapshot)
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()
        if self.available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
                
    def get_metrics_df(self):
        if not self.records:
            return pd.DataFrame()
        
        data = []
        for r in self.records:
            row = {'timestamp': r['timestamp']}
            for i, util in enumerate(r['gpu_utils']):
                row[f'gpu_{i}_util'] = util
            for i, mem in enumerate(r['mem_utils']):
                row[f'gpu_{i}_mem'] = mem
            data.append(row)
        return pd.DataFrame(data)

def process_gpu_trace(input_csv, output_dir="lora_configs", random_seed=42, 
                     arrival_scale=1.0, month=None):
    np.random.seed(random_seed)

    df = pd.read_csv(input_csv)
    df = df[df['gpu_num'] > 0].copy()
    
    print(f"Total GPU jobs after filtering: {len(df)}")
    df['submit_time'] = pd.to_datetime(df['submit_time'])

    if month is not None:
        df = df[df['submit_time'].dt.month == month].copy()
        print(f"Filtered to month {month}: {len(df)} jobs remaining")
        
        if len(df) == 0:
            print(f"Warning: No jobs found for month {month}")
            return None

    earliest_time = df['submit_time'].min()
    df['relative_start_time'] = (df['submit_time'] - earliest_time).dt.total_seconds()
    df['relative_start_time'] = df['relative_start_time'] * arrival_scale
    df = df.sort_values('relative_start_time').reset_index(drop=True)

    def categorize_job(gpu_count):
        if gpu_count == 1:
            return 'Light'
        elif 2 <= gpu_count <= 32:
            return 'Medium'
        else:
            return 'Heavy'
    
    df['job_category'] = df['gpu_num'].apply(categorize_job)

    def scale_gpu_count(category):
        mapping = {
            'Light': 1,
            'Medium': 2,
            'Heavy': 4
        }
        return mapping[category]
    
    df['scaled_gpu_count'] = df['job_category'].apply(scale_gpu_count)

    def assign_lora_rank(category):
        ranks = {
            'Light': [4, 8],
            'Medium': [16, 32],
            'Heavy': [32, 64]
        }
        return np.random.choice(ranks[category])
    
    def assign_batch_size(category):
        batch_sizes = {
            'Light': [2, 4, 8],
            'Medium': [16, 32],
            'Heavy': [32, 64]
        }
        return np.random.choice(batch_sizes[category])

    def assign_num_epochs(category):
        epochs = {
            'Light': [5, 10],
            'Medium': [25, 50],
            'Heavy': [50, 100]
        }
        return np.random.choice(epochs[category])

    def assign_base_model():
        models = ['llama-3-8b', 'qwen-3-8b']
        return np.random.choice(models)
    
    df['lora_rank'] = df['job_category'].apply(assign_lora_rank)
    df['batch_size'] = df['job_category'].apply(assign_batch_size)
    df['num_epochs'] = df['job_category'].apply(assign_num_epochs)
    df['base_model'] = [assign_base_model() for _ in range(len(df))]

    df = create_job_groups(df)
    create_yaml_configs(df, output_dir)

    print("\n=== Processing Statistics ===")
    print(f"Total jobs processed: {len(df)}")
    print(f"Total groups created: {df['group_id'].nunique()}")
    print(f"Random seed: {random_seed}")
    print(f"Arrival scale: {arrival_scale}x")
    if month:
        print(f"Month filter: {month}")
    print(f"\nJob category distribution:")
    print(df['job_category'].value_counts().sort_index())
    print(f"\nScaled GPU count distribution:")
    print(df['scaled_gpu_count'].value_counts().sort_index())
    print(f"\nBase model distribution:")
    print(df['base_model'].value_counts().sort_index())
    print(f"\nLoRA rank distribution:")
    print(df['lora_rank'].value_counts().sort_index())
    print(f"\nBatch size distribution:")
    print(df['batch_size'].value_counts().sort_index())
    print(f"\nNum epochs distribution:")
    print(df['num_epochs'].value_counts().sort_index())
    print(f"\nGroup size distribution (adapters per group):")
    print(df.groupby('group_id')['job_id'].count().value_counts().sort_index())
    print(f"\nTime span: {df['relative_start_time'].max():.2f} seconds")
    print(f"            ({df['relative_start_time'].max()/3600:.2f} hours)")
    
    return df


def create_job_groups(df):    
    df = df.copy()
    df['group_id'] = -1
    group_counter = 0

    df_sorted = df.sort_values(['base_model', 'relative_start_time']).reset_index(drop=True)
    
    for base_model in df_sorted['base_model'].unique():
        model_jobs = df_sorted[df_sorted['base_model'] == base_model].index.tolist()
        
        current_group_jobs = []
        current_group_gpus = 0
        
        for idx in model_jobs:
            job_gpus = df_sorted.loc[idx, 'scaled_gpu_count']

            if current_group_gpus + job_gpus <= 4:
                current_group_jobs.append(idx)
                current_group_gpus += job_gpus
            else:
                if current_group_jobs:
                    for job_idx in current_group_jobs:
                        df_sorted.loc[job_idx, 'group_id'] = group_counter
                    group_counter += 1

                current_group_jobs = [idx]
                current_group_gpus = job_gpus

        if current_group_jobs:
            for job_idx in current_group_jobs:
                df_sorted.loc[job_idx, 'group_id'] = group_counter
            group_counter += 1

    group_stats = df_sorted.groupby('group_id').agg({
        'scaled_gpu_count': 'sum',
        'job_id': 'count'
    }).rename(columns={
        'scaled_gpu_count': 'group_total_gpus',
        'job_id': 'group_adapter_count'
    })
    
    df_sorted = df_sorted.merge(group_stats, on='group_id', how='left')
    df_sorted = df_sorted.sort_values('relative_start_time').reset_index(drop=True)
    
    return df_sorted


def create_yaml_configs(df, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Generating YAML Configurations ===")
    print(f"Output directory: {output_dir}")

    for group_id, group_df in df.groupby('group_id'):
        adapters = []
        tasks = []

        base_model = group_df.iloc[0]['base_model']
        total_gpus = int(group_df['scaled_gpu_count'].sum())
        num_adapters = len(group_df)

        for idx, job in group_df.iterrows():
            job_id = job['job_id']
            lora_name = f"lora_{job_id}"
            
            adapter = {
                'name': lora_name,
                'type': 'lora',
                'path': f'adapters/lora_sft_{job_id}',
                'optimizer': 'adamw',
                'lr': 1e-5,
                'r': int(job['lora_rank']),
                'alpha': int(job['lora_rank'] * 2),
                'dropout': 0.05,
                'target_modules': {
                    'q_proj': True,
                    'k_proj': True,
                    'v_proj': True,
                    'o_proj': True,
                    'gate_proj': True,
                    'down_proj': True,
                    'up_proj': True
                }
            }
            adapters.append(adapter)
            
            task = {
                'type': 'train',
                'name': f'task_{job_id}',
                'adapter': lora_name,
                'dataset': 'gsm8k',
                'batch_size': int(job['batch_size']),
                'mini_batch_size': int(job['batch_size']),
                'num_epochs': int(job['num_epochs']),
                'cutoff_len': 2048,
                'save_step': 999999999
            }
            tasks.append(task)

        config = {
            'dispatcher': {
                'name': 'default',
                'concurrency_num': num_adapters
            },
            'datasets': [
                {
                    'name': 'gsm8k',
                    'data': 'demo/gsm8k.json',
                    'prompt': 'demo/prompt.yaml',
                    'prompt_type': 'instruction',
                    'preprocess': 'shuffle'
                }
            ],
            'adapters': adapters,
            'tasks': tasks
        }

        output_file = os.path.join(output_dir, f"group_{group_id}.yaml")
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated {df['group_id'].nunique()} YAML configuration files (one per group)")


def get_model_type(base_model):
    """Extract model type from base model name."""
    if 'llama' in base_model.lower():
        return 'llama'
    elif 'qwen' in base_model.lower():
        return 'qwen'
    else:
        return 'llama'


def get_full_model_path(base_model):
    if 'llama' in base_model.lower():
        return 'meta-llama/Llama-3.1-8B'
    elif 'qwen' in base_model.lower():
        return 'Qwen/Qwen3-8B'
    else:
        return 'meta-llama/Llama-3.1-8B'


def get_balance_config(base_model, total_gpus):    
    total_layers = 39 if base_model == "Qwen/Qwen3-8B" else 35

    layers_per_gpu = total_layers // total_gpus
    remainder = total_layers % total_gpus

    balance = [
        layers_per_gpu + (1 if i < remainder else 0)
        for i in range(total_gpus)
    ]

    return ' '.join(map(str, balance))


class WorkloadExecutor:    
    def __init__(self, df, config_dir, total_gpus=4):
        self.df = df
        self.config_dir = config_dir
        self.total_gpus = total_gpus

        self.job_metrics = []
        self.gpu_usage_timeline = []
        self.simulation_time = 0.0
        self.wall_clock_start = None

        self.monitor = GPUMonitor(interval=1.0)
        
    def generate_command(self, group_df, config_path):
        total_gpus = int(group_df['scaled_gpu_count'].sum())
        base_model = group_df.iloc[0]['base_model']
        model_type = get_model_type(base_model)
        full_model_path = get_full_model_path(base_model)
        balance = get_balance_config(base_model, total_gpus)
        
        command = [
            'torchrun',
            f'--nproc_per_node={total_gpus}',
            'mlora_pp_train.py',
            '--base_model', full_model_path,
            '--config', config_path,
            '--pipeline',
            '--device', 'cuda:',
            '--recompute',
            '--balance'
        ]

        command.extend(balance.split())

        command.extend([
            '--precision', 'fp16',
            '--model_type', model_type
        ])
        
        return command
    
    def execute_job(self, group_id, group_df, scheduled_time):
        if scheduled_time > self.simulation_time:
            time_skip = scheduled_time - self.simulation_time
            print(f"\n[Simulated time skip: {time_skip:.2f}s]")
            self.simulation_time = scheduled_time

        actual_start = self.simulation_time
        gpus_used = int(group_df['scaled_gpu_count'].sum())
        
        print(f"\n[t={actual_start:.2f}s] Starting Group {group_id}")
        print(f"  Model: {group_df.iloc[0]['base_model']}")
        print(f"  Adapters: {len(group_df)}")
        print(f"  GPUs: {gpus_used}")

        config_path = os.path.join(self.config_dir, f"group_{group_id}.yaml")
        command = self.generate_command(group_df, config_path)

        wall_start = time.time()
        
        try:
            log_dir = os.path.join(self.config_dir, "logs")
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_file = os.path.join(log_dir, f"group_{group_id}.log")

            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    start_new_session=True
                )
                
                print(f"  Logging to: {log_file}")
                print(f"  {'='*50}")
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ)

                try:
                    while True:
                        events = selector.select(timeout=180)

                        if not events:
                            print(f"\n  [!] No output for 180 seconds. Killing process group_{group_id}...")
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                            break

                        if events:
                            line = process.stdout.readline()
                            if not line:
                                break
                            f.write(line)
                            f.flush()
                            print(f"  [{group_id}] {line.rstrip()}")

                finally:
                    selector.close()
                    
                    cmd = "nvidia-smi --query-compute-apps=pid --format=csv,noheader"
                    output = subprocess.check_output(cmd, shell=True, encoding='utf-8')

                    pids_str = output.strip().split('\n')
                    pids = [int(p.strip()) for p in pids_str if p.strip()]

                    print(f"Found PIDs: {pids}")

                    for pid in pids:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except:
                            pass

                    process.wait()
                    os.killpg(process.pid, signal.SIGKILL)

            wall_completion_time = time.time() - wall_start

            self.simulation_time += wall_completion_time
            actual_end = self.simulation_time
            
            print(f"  {'='*50}")
            print(f"[t={actual_end:.2f}s] Completed Group {group_id} (duration: {wall_completion_time:.2f}s)")

            self.job_metrics.append({
                'group_id': group_id,
                'scheduled_time': scheduled_time,
                'actual_start': actual_start,
                'actual_end': actual_end,
                'completion_time': wall_completion_time,
                'gpus_used': gpus_used,
                'num_adapters': len(group_df),
                'base_model': group_df.iloc[0]['base_model'],
                'status': 'completed' if process.returncode == 0 else 'failed',
                'return_code': process.returncode,
                'log_file': log_file
            })
            
            if process.returncode != 0:
                print(f"  ERROR: Job failed with return code {process.returncode}")
                print(f"  Check log file: {log_file}")
            
        except Exception as e:
            print(f"  ERROR executing group {group_id}: {e}")
            import traceback
            traceback.print_exc()

            wall_completion_time = time.time() - wall_start
            self.simulation_time += wall_completion_time
            actual_end = self.simulation_time
            
            self.job_metrics.append({
                'group_id': group_id,
                'scheduled_time': scheduled_time,
                'actual_start': actual_start,
                'actual_end': actual_end,
                'completion_time': wall_completion_time,
                'gpus_used': gpus_used,
                'num_adapters': len(group_df),
                'base_model': group_df.iloc[0]['base_model'],
                'status': 'error',
                'return_code': -1,
                'log_file': None
            })
    
    def execute_workload(self):
        print("\n" + "="*60)
        print("STARTING WORKLOAD EXECUTION")
        print("="*60)
        
        self.wall_clock_start = time.time()
        self.simulation_time = 0.0

        group_info = self.df.groupby('group_id').agg({
            'relative_start_time': 'min',
            'base_model': 'first',
            'scaled_gpu_count': 'sum'
        }).reset_index()
        group_info = group_info.sort_values('relative_start_time')

        self.monitor.start()
        for _, group in group_info.iterrows():
            group_id = group['group_id']
            scheduled_time = group['relative_start_time']
            group_df = self.df[self.df['group_id'] == group_id]
            
            self.execute_job(group_id, group_df, scheduled_time)

        self.monitor.stop()
        self.monitor.join()
        
        wall_clock_end = time.time()
        total_wall_time = wall_clock_end - self.wall_clock_start
        
        print("\n" + "="*60)
        print("WORKLOAD EXECUTION COMPLETED")
        print("="*60)
        print(f"Simulated time: {self.simulation_time:.2f}s ({self.simulation_time/3600:.2f} hours)")
        print(f"Wall clock time: {total_wall_time:.2f}s ({total_wall_time/3600:.2f} hours)")
        print(f"Speedup: {self.simulation_time/total_wall_time:.2f}x")
        self.calculate_metrics()
    
    def calculate_metrics(self):
        if not self.job_metrics:
            print("No metrics to report.")
            return
        
        metrics_df = pd.DataFrame(self.job_metrics)
        
        total_simulated_time = self.simulation_time
        total_wall_time = time.time() - self.wall_clock_start
        completed_jobs = metrics_df[metrics_df['status'] == 'completed']

        hw_metrics = self.monitor.get_metrics_df()
        
        print("\n" + "="*60)
        print("GPU UTILIZATION")
        print("="*60)
        
        if not hw_metrics.empty:
            gpu_cols = [c for c in hw_metrics.columns if 'gpu_' in c and '_util' in c]

            avg_all_gpus = hw_metrics[gpu_cols].mean().mean()
            peak_all_gpus = hw_metrics[gpu_cols].max().max()
            
            print(f"Overall GPU Utilization: {avg_all_gpus:.2f}%")
            print(f"Peak GPU Utilization:    {peak_all_gpus:.2f}%")
            
            print("\nPer-GPU Statistics:")
            for col in gpu_cols:
                gpu_id = col.split('_')[1]
                avg = hw_metrics[col].mean()
                peak = hw_metrics[col].max()
                print(f"  GPU {gpu_id}: Avg {avg:.2f}%, Peak {peak:.2f}%")

            hw_trace_path = os.path.join(self.config_dir, "hardware_metrics.csv")
            hw_metrics.to_csv(hw_trace_path, index=False)
            print(f"\nHardware trace saved to: {hw_trace_path}")
        
        print("\n" + "="*60)
        print("EXECUTION METRICS")
        print("="*60)

        print(f"\nJob Completion:")
        print(f"  Total jobs: {len(metrics_df)}")
        print(f"  Completed: {len(completed_jobs)}")
        print(f"  Failed: {len(metrics_df[metrics_df['status'] == 'failed'])}")
        print(f"  Errors: {len(metrics_df[metrics_df['status'] == 'error'])}")
        
        if len(completed_jobs) > 0:
            print(f"\nCompletion Time Statistics:")
            print(f"  Mean: {completed_jobs['completion_time'].mean():.2f}s")
            print(f"  Median: {completed_jobs['completion_time'].median():.2f}s")
            print(f"  Min: {completed_jobs['completion_time'].min():.2f}s")
            print(f"  Max: {completed_jobs['completion_time'].max():.2f}s")
            print(f"  Std Dev: {completed_jobs['completion_time'].std():.2f}s")

        print(f"\nThroughput (Simulated Time):")
        print(f"  Total simulated time: {total_simulated_time:.2f}s ({total_simulated_time/3600:.2f} hours)")
        if total_simulated_time > 0:
            print(f"  Jobs per hour: {len(completed_jobs) / (total_simulated_time/3600):.2f}")
            print(f"  Adapters per hour: {completed_jobs['num_adapters'].sum() / (total_simulated_time/3600):.2f}")

        print(f"\nThroughput (Wall Clock Time):")
        print(f"  Total wall clock time: {total_wall_time:.2f}s ({total_wall_time/3600:.2f} hours)")
        if total_wall_time > 0:
            print(f"  Jobs per hour: {len(completed_jobs) / (total_wall_time/3600):.2f}")
            print(f"  Adapters per hour: {completed_jobs['num_adapters'].sum() / (total_wall_time/3600):.2f}")

        total_job_time = completed_jobs['completion_time'].sum()
        idle_time = total_simulated_time - total_job_time
        if total_simulated_time > 0:
            print(f"  Total job execution time: {total_job_time:.2f}s")
            print(f"  Total idle time: {idle_time:.2f}s ({idle_time/total_simulated_time*100:.2f}%)")

        if len(completed_jobs) > 0:
            print(f"\nBy Job Category:")
            category_stats = self.df.merge(
                metrics_df[['group_id', 'completion_time']], 
                on='group_id'
            )
            for category in ['Light', 'Medium', 'Heavy']:
                cat_jobs = category_stats[category_stats['job_category'] == category]
                if len(cat_jobs) > 0:
                    print(f"  {category}:")
                    print(f"    Count: {len(cat_jobs)}")
                    print(f"    Avg completion: {cat_jobs['completion_time'].mean():.2f}s")

        total_time_skipped = 0
        for i, row in metrics_df.iterrows():
            if row['scheduled_time'] > (metrics_df.iloc[i-1]['actual_end'] if i > 0 else 0):
                time_skipped = row['scheduled_time'] - (metrics_df.iloc[i-1]['actual_end'] if i > 0 else 0)
                total_time_skipped += time_skipped
        
        print(f"\nTime Skip Analysis:")
        print(f"  Total time skipped: {total_time_skipped:.2f}s ({total_time_skipped/3600:.2f} hours)")
        if total_simulated_time > 0:
            print(f"  Percentage of simulated time: {total_time_skipped/total_simulated_time*100:.2f}%")

        metrics_path = os.path.join(self.config_dir, "execution_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nDetailed metrics saved to: {metrics_path}")

if __name__ == "__main__":
    input_file = "trace_seren.csv"
    config_dir = "demo/lora"
    seed = 42

    arrival_scale = 1.0  # 0.5 for 2x faster, 2.0 for 2x slower
    month = 3  # Set to None for all months, 1-12 for specific month
    
    processed_df = process_gpu_trace(
        input_file, 
        output_dir=config_dir,
        random_seed=seed,
        arrival_scale=arrival_scale,
        month=month
    )
    
    if processed_df is not None:
        print(f"\nYAML configs saved to: {config_dir}/")
        print("\n" + "="*60)
        print("Ready to execute workload")
        print("="*60)

        executor = WorkloadExecutor(processed_df, config_dir, total_gpus=4)
        executor.execute_workload()