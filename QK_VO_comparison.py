import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime

class ExperimentAnalyzer:
    def __init__(self, log_dir="experiment_logs"):
        self.log_dir = Path(log_dir)
        self.output_dir = Path("experiment_results")
        self.output_dir.mkdir(exist_ok=True)
        
    def parse_logs(self):
        """Parse mLoRA logs for each experiment"""
        metrics = {
            'qk_only': {'loss': [], 'chosen_reward': [], 'rejected_reward': [], 'steps': []},
            'vo_only': {'loss': [], 'chosen_reward': [], 'rejected_reward': [], 'steps': []},
            'combined': {'loss': [], 'chosen_reward': [], 'rejected_reward': [], 'steps': []}
        }
        
        for log_file in self.log_dir.glob("*.log"):
            with open(log_file) as f:
                for line in f:
                    if "Task - exp1" in line and "loss:" in line:
                        # Parse the experiment type
                        if "qk_only" in line:
                            exp_type = 'qk_only'
                        elif "vo_only" in line:
                            exp_type = 'vo_only'
                        elif "combined" in line:
                            exp_type = 'combined'
                        else:
                            continue
                            
                        # Extract metrics
                        try:
                            loss = float(line.split("loss: ")[1].split(",")[0])
                            chosen = float(line.split("chosen_rewards: ")[1].split(",")[0])
                            rejected = float(line.split("rejected_rewards: ")[1].split()[0])
                            
                            metrics[exp_type]['loss'].append(loss)
                            metrics[exp_type]['chosen_reward'].append(chosen)
                            metrics[exp_type]['rejected_reward'].append(rejected)
                            metrics[exp_type]['steps'].append(len(metrics[exp_type]['loss']))
                        except:
                            continue
                            
        return metrics

    def generate_plots(self, metrics):
        """Generate comparison plots"""
        # Training Loss
        plt.figure(figsize=(12, 6))
        for exp_type in metrics:
            plt.plot(metrics[exp_type]['steps'], 
                    metrics[exp_type]['loss'], 
                    label=f'{exp_type}')
        plt.title('Training Loss Comparison')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.output_dir / 'loss_comparison.png')
        plt.close()
        
        # Reward Comparison
        plt.figure(figsize=(12, 6))
        for exp_type in metrics:
            plt.plot(metrics[exp_type]['steps'], 
                    metrics[exp_type]['chosen_reward'], 
                    label=f'{exp_type} (chosen)')
            plt.plot(metrics[exp_type]['steps'], 
                    metrics[exp_type]['rejected_reward'], 
                    label=f'{exp_type} (rejected)', 
                    linestyle='--')
        plt.title('Reward Comparison')
        plt.xlabel('Training Steps')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig(self.output_dir / 'reward_comparison.png')
        plt.close()

    def generate_summary(self, metrics):
        """Generate summary statistics"""
        summary = {}
        for exp_type in metrics:
            summary[exp_type] = {
                'final_loss': metrics[exp_type]['loss'][-1],
                'mean_loss': np.mean(metrics[exp_type]['loss']),
                'final_chosen_reward': metrics[exp_type]['chosen_reward'][-1],
                'final_rejected_reward': metrics[exp_type]['rejected_reward'][-1],
                'mean_chosen_reward': np.mean(metrics[exp_type]['chosen_reward']),
                'mean_rejected_reward': np.mean(metrics[exp_type]['rejected_reward']),
                'total_steps': len(metrics[exp_type]['steps'])
            }
        
        return pd.DataFrame(summary).T

    def create_report(self):
        """Create a comprehensive experiment report"""
        metrics = self.parse_logs()
        self.generate_plots(metrics)
        summary = self.generate_summary(metrics)
        
        # Save summary statistics
        summary.to_csv(self.output_dir / 'summary_stats.csv')
        
        # Create markdown report
        with open(self.output_dir / 'experiment_report.md', 'w') as f:
            f.write('# Heterogeneous LoRA Experiment Results\n\n')
            f.write(f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            
            f.write('## Summary Statistics\n\n')
            f.write(summary.to_markdown())
            
            f.write('\n\n## Key Findings\n\n')
            
            # Add key findings
            best_model = summary['mean_loss'].idxmin()
            best_reward = summary['mean_chosen_reward'].idxmax()
            
            f.write(f'- Best performing model (lowest loss): {best_model}\n')
            f.write(f'- Best reward model: {best_reward}\n')
            
            f.write('\n\n## Visualizations\n\n')
            f.write('### Training Loss Comparison\n')
            f.write('![Training Loss](loss_comparison.png)\n\n')
            f.write('### Reward Comparison\n')
            f.write('![Rewards](reward_comparison.png)\n')

if __name__ == "__main__":
    analyzer = ExperimentAnalyzer()
    analyzer.create_report()