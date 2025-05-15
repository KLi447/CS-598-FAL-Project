import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime

class PPOExperimentAnalyzer:
    def __init__(self, log_dir="experiment_logs"):
        self.log_dir = Path(log_dir)
        self.output_dir = Path("presentation_results")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_tensorboard_data(self):
        """Load data from TensorBoard logs"""
        experiments = {}
        for exp_dir in self.log_dir.glob("*"):
            if not exp_dir.is_dir():
                continue
                
            exp_name = exp_dir.name
            experiments[exp_name] = {
                'reward_loss': [],
                'actor_loss': [],
                'critic_loss': [],
                'policy_kl': [],
                'value_estimates': [],
                'rewards': [],
                'advantages': [],
                'gpu_memory': [],
                'training_time': [],
                'steps': []
            }
            
            for e in tf.compat.v1.train.summary_iterator(str(exp_dir / "events.*")):
                for v in e.summary.value:
                    if v.tag in experiments[exp_name]:
                        experiments[exp_name][v.tag].append((e.step, v.simple_value))
                        
        return experiments

    def generate_training_plots(self, experiments):
        """Generate plots for training metrics"""
        metrics = [
            'reward_loss', 'actor_loss', 'critic_loss',
            'policy_kl', 'value_estimates', 'rewards'
        ]
        
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            for exp_name, data in experiments.items():
                steps, values = zip(*data[metric])
                plt.plot(steps, values, label=exp_name)
            
            plt.title(f'{metric.replace("_", " ").title()} Over Training')
            plt.xlabel('Training Steps')
            plt.ylabel(metric.replace("_", " ").title())
            plt.legend()
            plt.grid(True)
            plt.savefig(self.output_dir / f'{metric}_comparison.png')
            plt.close()

    def generate_performance_plots(self, experiments):
        """Generate plots for performance metrics"""
        metrics = ['gpu_memory', 'training_time']
        
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            for exp_name, data in experiments.items():
                steps, values = zip(*data[metric])
                plt.plot(steps, values, label=exp_name)
            
            plt.title(f'{metric.replace("_", " ").title()} Usage')
            plt.xlabel('Training Steps')
            plt.ylabel('Memory (GB)' if 'memory' in metric else 'Time (s)')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.output_dir / f'{metric}_comparison.png')
            plt.close()

    def generate_summary_statistics(self, experiments):
        """Generate summary statistics for each experiment"""
        summary = {}
        for exp_name, data in experiments.items():
            summary[exp_name] = {
                'avg_reward_loss': np.mean([v for _, v in data['reward_loss']]),
                'avg_actor_loss': np.mean([v for _, v in data['actor_loss']]),
                'avg_critic_loss': np.mean([v for _, v in data['critic_loss']]),
                'final_reward': data['rewards'][-1][1],
                'peak_memory_gb': max([v for _, v in data['gpu_memory']]),
                'avg_step_time': np.mean([v for _, v in data['training_time']])
            }
        
        return pd.DataFrame(summary).T

    def create_presentation_report(self):
        """Create a comprehensive report for the presentation"""
        experiments = self.load_tensorboard_data()
        self.generate_training_plots(experiments)
        self.generate_performance_plots(experiments)
        summary = self.generate_summary_statistics(experiments)
        
        # Save summary statistics
        summary.to_csv(self.output_dir / 'experiment_summary.csv')
        
        # Create markdown report
        with open(self.output_dir / 'presentation_report.md', 'w') as f:
            f.write('# PPO Adapter Training Analysis\n\n')
            f.write(f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            
            f.write('## Summary Statistics\n\n')
            f.write(summary.to_markdown())
            
            f.write('\n\n## Training Metrics\n\n')
            f.write('### Loss Comparisons\n')
            f.write('![Reward Loss](reward_loss_comparison.png)\n')
            f.write('![Actor Loss](actor_loss_comparison.png)\n')
            f.write('![Critic Loss](critic_loss_comparison.png)\n')
            
            f.write('\n### Performance Metrics\n')
            f.write('![Policy KL](policy_kl_comparison.png)\n')
            f.write('![Value Estimates](value_estimates_comparison.png)\n')
            f.write('![Rewards](rewards_comparison.png)\n')
            
            f.write('\n### Resource Usage\n')
            f.write('![GPU Memory](gpu_memory_comparison.png)\n')
            f.write('![Training Time](training_time_comparison.png)\n')
            
            # Add key findings
            f.write('\n## Key Findings\n\n')
            best_reward = summary['final_reward'].idxmax()
            fastest = summary['avg_step_time'].idxmin()
            most_efficient = summary['peak_memory_gb'].idxmin()
            
            f.write(f'- Best Final Reward: {best_reward}\n')
            f.write(f'- Fastest Training: {fastest}\n')
            f.write(f'- Most Memory Efficient: {most_efficient}\n')

if __name__ == "__main__":
    analyzer = PPOExperimentAnalyzer()
    analyzer.create_presentation_report()