import mlora.model
import mlora.utils
import mlora.executor
import mlora.config
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    # Setup logging
    setup_logging()
    
    # Set random seed for reproducibility
    mlora.utils.setup_seed(42)
    
    # Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    args = mlora.utils.get_cmd_args()
    tokenizer, model = mlora.model.load_model(args)
    
    # Load configuration
    logging.info("Loading configuration...")
    config = mlora.config.MLoRAConfig("patent_classification.yaml")
    
    # Initialize executor
    logging.info("Initializing executor...")
    executor = mlora.executor.Executor(model, tokenizer, config)
    
    # Add task
    logging.info("Adding classification task...")
    for item in config.tasks_:
        executor.add_task(item)
    
    # Execute training
    logging.info("Starting training...")
    executor.execute()
    
    logging.info("Training complete!")

if __name__ == "__main__":
    main() 

#python train_patent.py --config patent_classification.yaml