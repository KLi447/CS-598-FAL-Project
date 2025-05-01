from datasets import load_dataset
import json
import os

# Load the dataset from Hugging Face
dataset = load_dataset("ccdv/patent-classification", "patent")

# Create a directory for the dataset if it doesn't exist
os.makedirs("data/patent", exist_ok=True)

# Format the data for training
def format_data(split_data):
    formatted_data = []
    for item in split_data:
        formatted_data.append({
            "text": item["text"],
            "label": item["label"]
        })
    return formatted_data

# Save the training split
with open("data/patent/train.json", "w") as f:
    json.dump({"data_points": format_data(dataset["train"])}, f)

# Save the validation split
with open("data/patent/validation.json", "w") as f:
    json.dump({"data_points": format_data(dataset["validation"])}, f)

# Save the test split
with open("data/patent/test.json", "w") as f:
    json.dump({"data_points": format_data(dataset["test"])}, f)

print("Dataset downloaded and saved to data/patent/")
print(f"Training samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['validation'])}")
print(f"Test samples: {len(dataset['test'])}") 