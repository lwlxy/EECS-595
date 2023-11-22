from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "cpu"
model_name_or_path = "meta-llama/Llama-2-7b-hf"
tokenizer_name_or_path = "meta-llama/Llama-2-7b-hf"
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Create other options for this question-answer pair:",
    tokenizer_name_or_path=model_name_or_path,
)

dataset_name = "twitter_complaints"
checkpoint_name = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace(
    "/", "_"
)
text_column = "Tweet text"
label_column = "text_label"
max_length = 64
lr = 3e-2
num_epochs = 50
batch_size = 8

# create results folder if doesn't exist
import os
if not os.path.exists("./results"):
    os.makedirs("./results")

# Prepare and tokenize dataset
train_dataset = load_dataset("json", data_files="train_dataset_processed.json")
val_dataset = load_dataset("json", data_files="val_dataset_processed.json")
test_dataset = load_dataset("json", data_files="test_dataset_processed.json")
tokenizer = AutoTokenizer.from_pretrained("t5-large")
prefix = "Create other options for this question-answer pair: "

print(train_dataset['train'])
# tokenized_train = train_dataset['train'].map(preprocess_function, batched=True)
# tokenized_val = val_dataset['train'].map(preprocess_function, batched=True)
# tokenized_test = test_dataset['train'].map(preprocess_function, batched=True)
# print(tokenized_test['input_ids'])

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token="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")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["question"]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(text_target=examples["options"])
    # model_inputs["labels"] = labels["input_ids"]
    # return model_inputs

    batch_size = len(examples["question"])

    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset['train'].map(
    preprocess_function,
    batched=True,
    num_proc=1,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)
val_dataset = val_dataset['train'].map(
    preprocess_function,
    batched=True,
    num_proc=1,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)
test_dataset = test_dataset['train'].map(
    preprocess_function,
    batched=True,
    num_proc=1,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)
print(test_dataset)

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(val_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, token="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")
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
model.save_pretrained("/model", from_pt=True)

with torch.no_grad():
    # inputs = {k: v.to(device) for k, v in test_dataset.items()}
    # outputs = model.generate(
    #     input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
    # )
    # print(tokenizer.batch_decode(outputs.detach().numpy(), skip_special_tokens=True))

    decoded_preds = []
    for i in range(len(test_dataset["input_ids"])):
        pred = torch.tensor([test_dataset["input_ids"][i]], device=device)
        print(pred)
        print(torch.tensor([test_dataset["attention_mask"][i]]))
        prediction = model.generate(input_ids=pred, attention_mask=torch.tensor([test_dataset["attention_mask"][i]], device=device))
        decoded_pred = tokenizer.batch_decode(prediction, skip_special_tokens=True)
        # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # rougeLSum expects newline after each sentence
        print(decoded_pred)
        decoded_preds.append("\n".join(decoded_pred))
        # decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in d$

        # write to file
    with open("./results/decoded_preds_peft.txt", "w") as f:
        f.write("\n".join(decoded_preds))
