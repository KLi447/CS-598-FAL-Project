import random
import json
from datasets import load_dataset

dataset = load_dataset("gsm8k", "main", split="train")

def paraphrase_instruction(question):
    templates = [
        "Can you solve this problem: {}",
        "What's the answer to this: {}",
        "Figure this out: {}",
        "Try solving: {}",
        "Answer this: {}"
    ]
    return random.choice(templates).format(question.strip())

entries = []
for example in dataset:
    question = example["question"]
    answer = example["answer"].split("####")[-1].strip()

    try:
        correct_number = float(answer.replace(",", ""))
        wrong_answer = str(int(correct_number + random.randint(1, 10)))
    except:
        wrong_answer = "42"

    entry = {
        "instruction": question.strip(),
        "instruction_paraphrased": paraphrase_instruction(question),
        "chosen": answer,
        "reject": wrong_answer
    }

    entries.append(entry)

with open("gsm8k.json", "w") as f:
    for entry in entries[:128]:
        f.write(json.dumps(entry) + "\n")

print(f"Saved {len(entries)} entries to gsm8k.json")