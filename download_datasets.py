from datasets import load_dataset

# Downloads and caches all splits (train, validation, test)
dataset = load_dataset("thu-coai/esconv")

# Inspect the structure and first training example
print(dataset)
print(dataset["train"][0])

dataset["train"].to_json("esconv_train.json")
# Or export to CSV:
# dataset["train"].to_csv("esconv_train.csv", index=False)