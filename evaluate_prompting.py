import json
import torch
import nltk
from datasets import load_dataset, load_metric

metric = load_metric("bleurt")

def compute_metric(preds, labels):
    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return result

with open("prompt_output_dataset.json") as f:
    dataset = json.load(f)
    new_dataset = []
    predictions = []
    labels = []
    for data in dataset:
        distractors = data['distractors']
        incorrect_answers = data['incorrect_answers']
        # print(distractors)
        # print(incorrect_answers)
        score_correspondences = []
        scores = []
        for d in distractors:
        #     results = []
        #     for label in incorrect_answers:
        #         results.append(compute_metric(d, label))
            results = compute_metric([d]*len(incorrect_answers), incorrect_answers)['scores']
            # print(len(results))
            if len(results) > 1:
                results = torch.tensor(results)
                value = torch.max(results).item()
                index = torch.argmax(results).item()
            else:
                value = results[0]
                index = 0
            score_correspondences.append(incorrect_answers[index])
            scores.append(value)
        data['score_correspondences'] = score_correspondences
        data['scores'] = scores
        new_dataset.append(data)
        print(data)
    json_string = json.dumps(new_dataset, indent=4)
    with open("prompt_output_dataset_bleurt.json", "w") as outfile:
        outfile.write(json_string)




