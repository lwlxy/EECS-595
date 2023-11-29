from datasets import Dataset
from datasets import load_dataset

train_dataset = load_dataset("json", data_files="train_dataset.json")['train']
val_dataset = load_dataset("json", data_files="val_dataset.json")['train']
test_dataset = load_dataset("json", data_files="test_dataset.json")['train']

questions = []
options = []

examples = train_dataset
for i in range(0, len(examples["best_answer"])):
    question = examples["question"][i] + " [SEP] " + examples["best_answer"][i]
    for s in examples["incorrect_answers"][i]:
        questions.append(question)
        options.append(s)
ds = Dataset.from_dict({"question": questions, "options": options})
ds.to_json("train_dataset_processed.json")
print(ds)

questions = []
options = []

examples = val_dataset
for i in range(0, len(examples["best_answer"])):
    question = examples["question"][i] + " [SEP] " + examples["best_answer"][i]
    for s in examples["incorrect_answers"][i]:
        questions.append(question)
        options.append(s)
ds = Dataset.from_dict({"question": questions, "options": options})
ds.to_json("val_dataset_processed.json")
print(ds)

questions = []
options = []

examples = test_dataset
for i in range(0, len(examples["best_answer"])):
    question = examples["question"][i] + " [SEP] " + examples["best_answer"][i] + '.'
    for s in examples["incorrect_answers"][i]:
        questions.append(question)
        options.append(s + '.')
ds = Dataset.from_dict({"question": questions, "options": options})
ds.to_json("test_dataset_processed.json")
print(ds)