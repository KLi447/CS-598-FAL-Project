import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'experiment_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def load_and_evaluate_model(base_model, adapter_path, test_examples, tokenizer):
    """Evaluate a specific adapter configuration"""
    model = PeftModel.from_pretrained(base_model, adapter_path)
    results = []
    
    for example in test_examples:
        # Generate response
        inputs = tokenizer(example["prompt"], return_tensors="pt")
        outputs = model.generate(**inputs, max_length=256)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Log results
        result = {
            "adapter": adapter_path,
            "prompt": example["prompt"],
            "response": response,
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }
        results.append(result)
        logging.info(f"Generated response for {adapter_path}: {response[:100]}...")
    
    return results

def main():
    # Load models and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.4")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.4")
    
    # Load test examples
    with open("data/hh-rlhf/test.json", 'r') as f:
        test_examples = json.load(f)[:10]  # Use 10 test examples
    
    # Configurations to test
    configs = [
        "adapters/exp1_lora_qk",
        "adapters/exp1_lora_vo"
    ]
    
    # Run evaluations
    all_results = []
    for config in configs:
        results = load_and_evaluate_model(base_model, config, test_examples, tokenizer)
        all_results.extend(results)
    
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(f'experiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    
    # Create visualizations
    plt.figure(figsize=(12, 6))
    # Add your visualization code here
    
    logging.info("Evaluation complete. Results saved.")

if __name__ == "__main__":
    main()
