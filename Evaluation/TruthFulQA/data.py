from datasets import load_dataset
import json

# Load the dataset
ds = load_dataset("domenicrosati/TruthfulQA")

# Initialize a dictionary to store data
result = {}

# For each split in the dataset (likely just 'train')
for split_name, split_dataset in ds.items():
    # Convert to list of dictionaries (each row becomes a dict)
    data_list = []
    for i in range(len(split_dataset)):
        data_list.append({k: split_dataset[i][k] for k in split_dataset[i].keys()})
    
    # Add this split to the result dictionary
    result[split_name] = data_list

# Save as JSON file
with open("TruthfulQA.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Dataset successfully saved to TruthfulQA.json")