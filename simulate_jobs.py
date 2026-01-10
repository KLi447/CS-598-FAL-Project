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
                     arrival_scale=1.0, month=None, max_jobs=None):
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

    if max_jobs and len(df) > max_jobs:
        df = df.iloc[:max_jobs].copy()
        print(f"Limiting simulation to first {max_jobs} jobs")

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
            'Light': [3, 5, 10],
            'Medium': [15, 25, 30],
            'Heavy': [50, 75, 100]
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
    
    print(f"Generated {df['group_id'].nunique()} YAML configuration files")


def get_model_type(base_model):
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
    total_layers = 39 if base_model == "qwen-3-8b" else 35

    layers_per_gpu = total_layers // total_gpus
    remainder = total_layers % total_gpus

    balance = [
        layers_per_gpu + (1 if i < remainder else 0)
        for i in range(total_gpus)
    ]

    return ' '.join(map(str, balance))


class WorkloadExecutor:    
    def __init__(self, df, config_dir, total_gpus=4, max_runtime_per_group=20):
        self.df = df
        self.config_dir = config_dir
        self.total_gpus = total_gpus
        self.max_runtime_per_group = max_runtime_per_group
        
        self.job_metrics = []
        self.simulation_time = 0.0
        self.wall_clock_start = None
        self.monitor = GPUMonitor(1.0)
        
    def create_minimal_config(self, group_df, config_path):
        adapters = []
        tasks = []
        
        base_model = group_df.iloc[0]['base_model']
        
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
                    'gate_proj': False,
                    'down_proj': False,
                    'up_proj': False
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
                'num_epochs': 1,
                'cutoff_len': 2048,
                'save_step': 999999999
            }
            tasks.append(task)
        
        config = {
            'dispatcher': {
                'name': 'default',
                'concurrency_num': len(group_df)
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
        
        output_file = os.path.join(config_path)
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def parse_log_timing(self, log_file, group_df):
        if not os.path.exists(log_file):
            return None
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        adapter_times = {}

        for idx, job in group_df.iterrows():
            job_id = job['job_id']
            lora_name = f"lora_{job_id}"

            pattern = rf'\[(\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}}),\d+\] m-LoRA: .*{lora_name}'
            matches = re.findall(pattern, log_content)
            
            if len(matches) >= 2:
                timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in matches]

                load_pattern = rf'\[(\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}}),\d+\] m-LoRA: Task to running, need to load adapters: \[.*{lora_name}.*\]'
                offload_pattern = rf'\[(\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}}),\d+\] m-LoRA: Finish and base model offload adapter - \[.*{lora_name}.*\]'
                
                load_match = re.search(load_pattern, log_content)
                offload_match = re.search(offload_pattern, log_content)
                
                if load_match and offload_match:
                    load_time = datetime.strptime(load_match.group(1), '%Y-%m-%d %H:%M:%S')
                    offload_time = datetime.strptime(offload_match.group(1), '%Y-%m-%d %H:%M:%S')

                    epoch_time = (offload_time - load_time).total_seconds()
                    adapter_times[job_id] = epoch_time
                    
                    print(f"  {lora_name}: {epoch_time:.2f}s per epoch")
        
        return adapter_times if adapter_times else None
    
    def estimate_group_runtime(self, group_df, adapter_times):
        max_time = 0
        
        for idx, job in group_df.iterrows():
            job_id = job['job_id']
            num_epochs = job['num_epochs']
            
            if job_id in adapter_times:
                total_time = adapter_times[job_id] * num_epochs * math.ceil(7473/128) # scale up by epochs and full dataset size
                max_time = max(max_time, total_time)
        
        return max_time
    
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
    
    def execute_calibration_run(self, group_id, group_df, scheduled_time):
        if scheduled_time > self.simulation_time:
            time_skip = scheduled_time - self.simulation_time
            print(f"\n[Simulated time skip: {time_skip:.2f}s]")
            self.simulation_time = scheduled_time
        
        actual_start = self.simulation_time
        gpus_used = int(group_df['scaled_gpu_count'].sum())
        
        print(f"\n{'='*60}")
        print(f"[t={actual_start:.2f}s] Starting Group {group_id}")
        print(f"  Model: {group_df.iloc[0]['base_model']}")
        print(f"  Adapters: {len(group_df)}")
        print(f"  GPUs: {gpus_used}")
        print(f"  Running calibration (1 epoch)...")

        config_path = os.path.join(self.config_dir, f"group_{group_id}_calibrate.yaml")
        self.create_minimal_config(group_df, config_path)
        
        command = self.generate_command(group_df, config_path)
        
        log_dir = os.path.join(self.config_dir, "logs")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = os.path.join(log_dir, f"group_{group_id}_calibrate.log")
        
        wall_start = time.time()
        
        try:
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
                
                import selectors
                import signal
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ)
                
                try:
                    while True:
                        events = selector.select(timeout=20)
                        
                        if not events:
                            print(f"\n  [!] No output for 20 seconds. Killing process group_{group_id}...")
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
            
            calibration_time = time.time() - wall_start
            print(f"  {'='*50}")
            print(f"  Calibration completed in {calibration_time:.2f}s")

            print(f"  Parsing timing information...")
            adapter_times = self.parse_log_timing(log_file, group_df)
            
            if adapter_times:
                estimated_runtime = self.estimate_group_runtime(group_df, adapter_times)
                print(f"  Estimated full runtime: {estimated_runtime:.2f}s")

                self.simulation_time += estimated_runtime
                actual_end = self.simulation_time
                
                print(f"  {'='*50}")
                print(f"[t={actual_end:.2f}s] Completed Group {group_id}")
                print(f"  Simulated duration: {estimated_runtime:.2f}s")
                print(f"  Wall clock time: {calibration_time:.2f}s")
                print(f"  Speedup: {estimated_runtime/calibration_time:.2f}x")

                self.job_metrics.append({
                    'group_id': group_id,
                    'scheduled_time': scheduled_time,
                    'actual_start': actual_start,
                    'actual_end': actual_end,
                    'completion_time': estimated_runtime,
                    'calibration_time': calibration_time,
                    'gpus_used': gpus_used,
                    'num_adapters': len(group_df),
                    'base_model': group_df.iloc[0]['base_model'],
                    'status': 'simulated',
                    'adapter_times': adapter_times,
                    'log_file': log_file
                })
                
            else:
                self.simulation_time += calibration_time
                
                self.job_metrics.append({
                    'group_id': group_id,
                    'scheduled_time': scheduled_time,
                    'actual_start': actual_start,
                    'actual_end': self.simulation_time,
                    'completion_time': calibration_time,
                    'calibration_time': calibration_time,
                    'gpus_used': gpus_used,
                    'num_adapters': len(group_df),
                    'base_model': group_df.iloc[0]['base_model'],
                    'status': 'fallback',
                    'log_file': log_file
                })
                
        except Exception as e:
            print(f"  ERROR executing group {group_id}: {e}")
            import traceback
            traceback.print_exc()
            
            wall_time = time.time() - wall_start
            self.simulation_time += wall_time
            
            self.job_metrics.append({
                'group_id': group_id,
                'scheduled_time': scheduled_time,
                'actual_start': actual_start,
                'actual_end': self.simulation_time,
                'completion_time': wall_time,
                'calibration_time': wall_time,
                'gpus_used': gpus_used,
                'num_adapters': len(group_df),
                'base_model': group_df.iloc[0]['base_model'],
                'status': 'error',
                'log_file': log_file
            })
    
    def execute_workload(self):
        print("\n" + "="*60)
        print("STARTING WORKLOAD SIMULATION")
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
            
            self.execute_calibration_run(group_id, group_df, scheduled_time)

        self.monitor.stop()
        self.monitor.join()
        
        wall_clock_end = time.time()
        total_wall_time = wall_clock_end - self.wall_clock_start
        
        print("\n" + "="*60)
        print("WORKLOAD SIMULATION COMPLETED")
        print("="*60)
        print(f"Simulated time: {self.simulation_time:.2f}s ({self.simulation_time/3600:.2f} hours)")
        print(f"Wall clock time: {total_wall_time:.2f}s ({total_wall_time/3600:.2f} hours)")
        print(f"Overall speedup: {self.simulation_time/total_wall_time:.2f}x")
        
        self.calculate_metrics()
    
    def calculate_metrics(self):
        if not self.job_metrics:
            print("No metrics to report.")
            return
        
        metrics_df = pd.DataFrame(self.job_metrics)
        
        total_simulated_time = self.simulation_time
        total_wall_time = time.time() - self.wall_clock_start

        hw_metrics = self.monitor.get_metrics_df()
        
        print("\n" + "="*60)
        print("HARDWARE UTILIZATION (Real-time)")
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
        print(f"  Total groups: {len(metrics_df)}")
        print(f"  Simulated: {len(metrics_df[metrics_df['status'] == 'simulated'])}")
        print(f"  Fallback: {len(metrics_df[metrics_df['status'] == 'fallback'])}")
        print(f"  Errors: {len(metrics_df[metrics_df['status'] == 'error'])}")
        
        completed = metrics_df[metrics_df['status'].isin(['simulated', 'fallback'])]
        
        if len(completed) > 0:
            print(f"\nSimulated Completion Time Statistics:")
            print(f"  Mean: {completed['completion_time'].mean():.2f}s")
            print(f"  Median: {completed['completion_time'].median():.2f}s")
            print(f"  Min: {completed['completion_time'].min():.2f}s")
            print(f"  Max: {completed['completion_time'].max():.2f}s")
            
            print(f"\nCalibration Time Statistics:")
            print(f"  Mean: {completed['calibration_time'].mean():.2f}s")
            print(f"  Median: {completed['calibration_time'].median():.2f}s")
            
            print(f"\nAverage Speedup per Group:")
            speedups = completed['completion_time'] / completed['calibration_time']
            print(f"  Mean: {speedups.mean():.2f}x")
            print(f"  Median: {speedups.median():.2f}x")
        
        print(f"\nThroughput (Simulated Time):")
        print(f"  Total simulated time: {total_simulated_time:.2f}s ({total_simulated_time/3600:.2f} hours)")
        if total_simulated_time > 0:
            print(f"  Groups per hour: {len(completed) / (total_simulated_time/3600):.2f}")
            print(f"  Adapters per hour: {completed['num_adapters'].sum() / (total_simulated_time/3600):.2f}")

        metrics_path = os.path.join(self.config_dir, "simulation_metrics.csv")
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
        month=month,
        max_jobs=1000
    )
    
    if processed_df is not None:
        print(f"\nYAML configs saved to: {config_dir}/")
        print("\n" + "="*60)
        print("Ready to execute workload")
        print("="*60)

        executor = WorkloadExecutor(processed_df, config_dir, total_gpus=4)
        executor.execute_workload()