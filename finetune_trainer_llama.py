import nltk
from datasets import load_dataset, load_metric
import evaluate
import numpy as np
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, BitsAndBytesConfig, TrainingArguments
from transformers import AutoModelForSeq2SeqLM, LlamaForCausalLM, LlamaTokenizer
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

# Prepare and tokenize dataset
train_dataset = load_dataset("json", data_files="train_dataset_processed.json")
val_dataset = load_dataset("json", data_files="val_dataset_processed.json")
test_dataset = load_dataset("json", data_files="test_dataset_processed.json")
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b", token="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")
prefix = "Create other options for this question-answer pair: "

def preprocess_function(examples):
    # inputs = [prefix + doc for doc in examples["question"]]
    # model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    #
    # labels = tokenizer(text_target=examples["options"], max_length=128, truncation=True)
    #
    # model_inputs["labels"] = labels["input_ids"]
    # return model_inputs
    output_texts = []

    for i in range(len(examples['question'])):
        text = f"Input: ```{examples['question'][i]}```\n Output: {examples['options'][i]}"
        output_texts.append(text)

    return output_texts

print(train_dataset['train'])
# tokenized_train = train_dataset['train'].map(preprocess_function, batched=True)
# tokenized_val = val_dataset['train'].map(preprocess_function, batched=True)
# tokenized_test = test_dataset['train'].map(preprocess_function, batched=True)
tokenized_train = train_dataset['train']
tokenized_val = val_dataset['train']
tokenized_test = test_dataset['train']

# Setup evaluation
nltk.download("punkt", quiet=True)
# metric = evaluate.load("bleurt")
metric = load_metric("bleurt")
# metric = load("bleurt", module_type="metric")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return result

# Load pretrained model and evaluate model after each epoch
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
bnb_config = BitsAndBytesConfig(
 load_in_4bit=True,
 bnb_4bit_quant_type="nf4",
 # bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token="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") #quantization_config=bnb_config,
peft_config = LoraConfig(
    r=128,
    lora_alpha=16,
    # target_modules=find_all_linear_names(base_model),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, peft_config)

# model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
# freeze everything
for name, param in model.named_parameters():
    print(name)
    param.requires_grad = False

# and Un-Freeze lower 4 layers of encoder
for i in range(0,4):
    for param in model.model.model.layers[i].parameters():
        param.requires_grad = True

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing =True,
    max_grad_norm= 0.3,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    logging_steps=100,
    eval_steps=100,
    # fp16=True,
    # predict_with_generate=True,
    warmup_ratio=0.05,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    # data_collator=data_collator,
    formatting_func=preprocess_function,
    compute_metrics=compute_metrics
)

trainer.train()
print(trainer.evaluate(tokenized_test))