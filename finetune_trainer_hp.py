import nltk
from datasets import load_dataset, load_metric
import evaluate
import numpy as np
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForCausalLM

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

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["question"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=examples["options"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print(train_dataset['train'])
tokenized_train = train_dataset['train'].map(preprocess_function, batched=True)
tokenized_val = val_dataset['train'].map(preprocess_function, batched=True)
tokenized_test = test_dataset['train'].map(preprocess_function, batched=True)

# Setup evaluation
nltk.download("punkt", quiet=True)
# metric = evaluate.load("bleurt")
metric = load_metric("bleurt")
# metric = load("bleurt", module_type="metric")

# Hyperparameter tuning!
# Change num_train_epochs, learning_rate, and weight_decay
num_train_epochs = [5, 10, 20]
learning_rate = [2e-4, 2e-5, 2e-6]
weight_decay = [0.1, 0.01, 0.001]

for epoch in num_train_epochs:
    for lr in learning_rate:
        for wd in weight_decay:
            print("num_train_epochs: ", epoch)
            print("learning_rate: ", lr)
            print("weight_decay: ", wd)

            file_name_prefix = "ep_" + str(epoch) + "_lr_" + str(lr) + "_wd_" + str(wd)

            def compute_metrics(eval_preds):
                preds, labels = eval_preds

                # decode preds and labels
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                # rougeLSum expects newline after each sentence
                decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
                decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

                # write to file
                with open(f"./results/{file_name_prefix}-decoded_preds.txt", "w") as f:
                    f.write("\n".join(decoded_preds))
                with open(f"./results/{file_name_prefix}-decoded_labels.txt", "w") as f:
                    f.write("\n".join(decoded_labels))

                result = metric.compute(predictions=decoded_preds, references=decoded_labels)
                return result

            # Load pretrained model and evaluate model after each epoch
            # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
            model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
            # freeze everything
            for param in model.parameters():
                param.requires_grad = False

            # and Un-Freeze lower 4 layers of encoder
            for i in range(0,4):
                for param in model.encoder.block[i].parameters():
                    param.requires_grad = True

            training_args = Seq2SeqTrainingArguments(
                output_dir="./results",
                evaluation_strategy="epoch",
                learning_rate=lr,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=4,
                weight_decay=wd,
                save_total_limit=3,
                num_train_epochs=epoch,
                logging_steps=100,
                eval_steps=100,
                # fp16=True,
                predict_with_generate=True
            )

            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics
            )

            trainer.train()

            print("Saving model...")
            trainer.save_model("./results")

            # Evaluate on test set and print results
            predictions = trainer.predict(tokenized_test)
            print(predictions.predictions)
            print(predictions.label_ids)
            print(predictions.metrics)

            # write metrics to file
            with open(f"./results/{file_name_prefix}-metrics.txt", "w") as f:
                f.write(str(predictions.metrics))

            # decode preds and labels
            labels = np.where(predictions.label_ids != -100, predictions.label_ids, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # write to file
            with open(f"./results/{file_name_prefix}-decoded_preds_2.txt", "w") as f:
                f.write("\n".join(decoded_preds))
            with open(f"./results/{file_name_prefix}-decoded_labels_2.txt", "w") as f:
                f.write("\n".join(decoded_labels))
