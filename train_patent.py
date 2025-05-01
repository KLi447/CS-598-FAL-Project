import mlora.model
import mlora.utils
import mlora.executor
import mlora.config
import logging
import argparse
from mlora.executor.task.dpo_task import DPOTask
from mlora.executor.task.task import TASK_CLASS

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    # Setup logging
    setup_logging()
    
    # Register DPO task type
    TASK_CLASS["dpo"] = DPOTask
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--precision", type=str, default="fp32", help="Precision to use")
    parser.add_argument("--model_type", type=str, default="llama", help="Type of model (llama)")
    parser.add_argument("--pipeline", action="store_true", help="Whether to use pipeline parallelism")
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    mlora.utils.setup_seed(42)
    
    # Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    tokenizer, model = mlora.model.load_model(args)
    
    # Load configuration
    logging.info("Loading configuration...")
    config = mlora.config.MLoRAConfig(args.config)
    
    # Initialize executor
    logging.info("Initializing executor...")
    executor = mlora.executor.Executor(model, tokenizer, config)
    
    # Add task
    logging.info("Adding DPO task...")
    for item in config.tasks_:
        executor.add_task(item)
    
    # Execute training
    logging.info("Starting DPO training...")
    executor.execute()
    
    logging.info("DPO training complete!")

if __name__ == "__main__":
    main() 

# Example command:
# python train_patent.py --base_model TinyLlama/TinyLlama-1.1B-Chat-v0.4 --config dpo_config.yaml --device cuda:0 --precision fp32