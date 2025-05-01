from datasets import load_dataset

# Download only 1000 examples from each split
train_dataset = load_dataset("Anthropic/hh-rlhf", split="train[:2500]")
test_dataset = load_dataset("Anthropic/hh-rlhf", split="test[:1500]")  # smaller test set

# Save to local JSON files
train_dataset.to_json("data/hh-rlhf/train.json")
test_dataset.to_json("data/hh-rlhf/test.json")