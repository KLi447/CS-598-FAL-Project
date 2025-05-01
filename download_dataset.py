from datasets import load_dataset

# Download the dataset
dataset = load_dataset("Anthropic/hh-rlhf")

# Save to local JSON files
dataset["train"][:3000].to_json("data/hh-rlhf/train.json")
dataset["test"][:1500].to_json("data/hh-rlhf/test.json")