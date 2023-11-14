import argparse
import pickle
import os
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, load_metric, Dataset
from tqdm.auto import tqdm


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def load_data(tokenizer, params):

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
    #                   TODO: Implementation                      #
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    def tokenize_function(examples):
        # Filter out data that does not have 4 options
        questions = []
        options = []
        labels = []
        for i in range(0, len(examples["best_answer"])):
            question = examples["question"][i] + examples["best_answer"][i]
            # label the correct_answers as bad distractors
            # for s in examples["correct_answers"][i]:
            #     # example["labels"][i] = examples["choices"][i]["label"].index(examples["answerKey"][i])
            #     labels.append(0)
            #     questions.append(question)
            #     options.append(s)
            # label the incorrect_answers as good distractors
            for s in examples["incorrect_answers"][i]:
                # labels.append(1)
                questions.append(question)
                options.append(s)
            # if len(examples["choices"][i]["text"]) != 4:
            #     del examples["choices"][i]
            #     del examples["question"][i]
            #     del examples["id"][i]
            #     del examples["answerKey"][i]
        # question = [[context] * 4 for context in examples["question"]]
        # options = [[t for t in example["text"]] for example in examples["choices"]]
        # Flatten everything
        # first_sentences = sum(questions, [])
        # second_sentences = sum(options, [])

        # Tokenize  padding="max_length",
        tokenized_questions = tokenizer(questions, truncation=True)
        tokenized_options = tokenizer(options, truncation=True)
        # examples["labels"] = labels
        # examples = examples.remove_columns(["type", "category", "source"])
        # print({k: [v[i] for i in range(0, len(v))] for k, v in tokenized_examples.items()})
        # print(len({k: [v[i] for i in range(0, len(v))] for k, v in tokenized_examples.items()}))
        # print(tokenized_examples.keys())

        # Un-flatten
        # return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        # tokenized_examples = tokenizer(, padding="max_length", truncation=True)
        return {"input_ids": tokenized_questions['input_ids'], "options": tokenized_options['input_ids']}

    from dataclasses import dataclass
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
    from typing import Optional, Union

    @dataclass
    class DataCollatorForMultipleChoice:
        """
        Data collator that will dynamically pad the inputs for multiple choice received.
        """

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None

        def __call__(self, features):

            labels = [feature.pop('options') for feature in features]
            batch_size = len(features)
            print(labels)
            # num_choices = 1
            # print(num_choices)

            # Flatten
            flattened_features = [[{k: v for k, v in feature.items()}] for feature in features]
            flattened_features = sum(flattened_features, [])
            print(flattened_features)

            # Apply Padding
            batch = self.tokenizer.pad(
                flattened_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            # Un-flatten
            batch = {k: v.view(batch_size, -1).to(device) for k, v in batch.items()}

            # Add back labels
            batch["options"] = torch.tensor(labels, device=device)

            return batch

    # split dataset
    # train_dataset = load_dataset('truthful_qa', 'generation', split='validation')
    # train_dataset = train_dataset.train_test_split(test_size=0.2)
    # train_dataset['train'] = train_dataset['train'].train_test_split(test_size=0.25)
    # train_dataset = load_dataset('truthful_qa', 'generation', split='validation[0%:60%](pct1_dropremainder)')
    # val_dataset = load_dataset('truthful_qa', 'generation', split='validation[60%:80%](pct1_dropremainder)')
    # test_dataset = load_dataset('truthful_qa', 'generation', split='validation[80%:100%](pct1_dropremainder)')
    # import pickle
    # print(train_dataset)
    # with open('train_dataset.pickle', 'wb') as output:
    #     pickle.dump(train_dataset['train']['train'], output)
    # with open('val_dataset.pickle', 'wb') as output:
    #     pickle.dump(train_dataset['train']['test'], output)
    # with open('test_dataset.pickle', 'wb') as output:
    #     pickle.dump(train_dataset['test'], output)

    with open('train_dataset.pickle', 'rb') as output:
        train_dataset = pickle.load(output)
    with open('val_dataset.pickle', 'rb') as output:
        val_dataset = pickle.load(output)
    with open('test_dataset.pickle', 'rb') as output:
        test_dataset = pickle.load(output)

    print(train_dataset)

    print(val_dataset)
    # tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_train_dataset = Dataset.from_dict(tokenize_function(train_dataset))
    print(tokenized_train_dataset)
    # tokenized_train_dataset = tokenized_train_dataset.remove_columns(["type", "category", "source"])
    # tokenized_dataset = tokenized_dataset.rename_column("answerKey", "labels")
    tokenized_train_dataset.set_format("torch")
    tokenized_val_dataset = Dataset.from_dict(tokenize_function(val_dataset))
    tokenized_val_dataset.set_format("torch")
    tokenized_test_dataset = Dataset.from_dict(tokenize_function(test_dataset))
    tokenized_test_dataset.set_format("torch")

    batch_size = params.batch_size
    # train_dataset = tokenized_datasets["train"].shuffle(seed=SEED).select(range(1000))
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    # eval_dataset = tokenized_datasets["validation"]
    # eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    # test_dataset = tokenized_datasets["test"]
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    # print(tokenizer.decode(train_dataset["input_ids"][0][0]))
    # print(tokenizer.decode(train_dataset["input_ids"][0][1]))

    data_collator = DataCollatorForMultipleChoice(tokenizer)
    # train_dataset = tokenized_datasets["train"].shuffle(seed=SEED)
    train_dataloader = DataLoader(tokenized_train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    # eval_dataset = tokenized_datasets["validation"]
    eval_dataloader = DataLoader(tokenized_val_dataset, batch_size=batch_size, collate_fn=data_collator)
    # test_dataset = tokenized_datasets["test"]
    test_dataloader = DataLoader(tokenized_test_dataset, batch_size=batch_size, collate_fn=data_collator)

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    return train_dataloader, eval_dataloader, test_dataloader


def finetune(model, train_dataloader, eval_dataloader, params):

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
    #                   TODO: Implementation                      #
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    import evaluate
    from torch.optim import AdamW
    from transformers import get_scheduler

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0.1 * params.num_epochs * len(train_dataloader),
        num_training_steps=params.num_epochs * len(train_dataloader)
    )
    # metric = evaluate.load("bleurt")
    metric = load_metric('bleurt')
    metric1 = load_metric('bleurt')

    for epoch in range(params.num_epochs):

        model.train()
        progress_bar = tqdm(range(len(train_dataloader)))
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            progress_bar.update(1)
            metric1.add_batch(predictions=predictions, references=batch["options"])

            optimizer.zero_grad()
        score = metric1.compute()
        print('Train Accuracy:', score)

        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        score = metric.compute()
        print('Loss:', loss.item(), 'Validation Accuracy:', score)
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    return model


def test(model, test_dataloader, prediction_save='predictions.torch'):

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
    #                   TODO: Implementation                      #
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    metric = load_metric('bleurt')
    model.eval()
    all_predictions = []

    for batch in test_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    score = metric.compute()
    print('Test Accuracy:', score)
    torch.save(all_predictions, prediction_save)

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #


def main(params):

    tokenizer = AutoTokenizer.from_pretrained(params.model)
    train_dataloader, eval_dataloader, test_dataloader = load_data(tokenizer, params)

    model = AutoModelForSeq2SeqLM.from_pretrained(params.model)
    model.to(device)
    for name, param in model.named_parameters():
        if not name.startswith("roberta.encoder.layer.23"): # choose whatever you like here
          param.requires_grad = False
    model = finetune(model, train_dataloader, eval_dataloader, params)

    test(model, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Language Model")

    parser.add_argument("--dataset", type=str, default="truthful_qa")
    parser.add_argument("--model", type=str, default="t5-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)

    params, unknown = parser.parse_known_args()
    main(params)
