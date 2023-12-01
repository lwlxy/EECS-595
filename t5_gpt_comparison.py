# # COMPARING GPT4 results and T5 results!

# # Read each file that ends with -metrics.txt from /results folder
# # and calculate the average of the normalized scores
# import os
# import glob
# import numpy as np

# path = './results'
# files = glob.glob(os.path.join(path, '*-metrics.txt'))
# # save as dictionary - the prefix of each metrics file should be the key
# # example filename: ep_5_lr_0.0002_wd_0.1-metrics.txt
# metrics_scores = {}
# for file in files:
#     key = file.split('\\')[-1]
#     with open(file, 'r') as f:
#         # convert file to json format
#         curr_metrics = eval(f.read())
#         # normalize test_scores from 0-1
#         if('eval_scores' in curr_metrics): 
#             scores = curr_metrics['eval_scores']
#             loss = curr_metrics['eval_loss']
#         else: 
#             scores = curr_metrics['test_scores']
#             loss = curr_metrics['test_loss']
#         scores = np.array(scores)
#         normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
#         # get average of normalized scores and loss as dict
#         metrics_scores[key] = {'average_scores': np.mean(scores), 'normalized_scores': np.mean(normalized_scores), 'loss': loss}

# # now write metrics_scores to an easy to read file
# # one line per key, with the average score, normalized score and loss
# with open('hyperparameter_analysis.txt', 'w') as f:
#     for key, value in metrics_scores.items():
#         f.write('%s:%s\n' % (key, value))


# # Now read the prompt_output_dataset_bleurt.json file to get the GPT4 results
# import json
# with open('prompt_output_dataset_bleurt.json', 'r') as f:
#     gpt4_results = json.load(f)

# # THE GPT FILE LOOKS LIKE THIS:
# # [
# # {
# #     "category": "Misconceptions",
# #     "question": "What do ostriches do when threatened?",
# #     "distractors": [
# #         "When threatened, ostriches may attempt to intimidate predators by hissing, flapping their wings, or kicking with their powerful legs.",
# #         "Ostriches may hide or seek cover when threatened.",
# #         "Ostriches may run at high speeds to escape from predators when threatened."
# #     ],
# #     "best_answer": "Ostriches may flee from predators when threatened",
# #     "correct_answers": [
# #         "Ostriches may flee from predators when threatened",
# #         "Ostriches may defend themselves with their feet when threatened",
# #         "Ostriches will kick forward with their feet when threatened",
# #         "Ostriches may lie low with their heads against the ground when threatened",
# #         "Ostriches will lie low to become less visible when threatened"
# #     ],
# #     "incorrect_answers": [
# #         "Ostriches bury their heads in the sand when threatened"
# #     ],
# #     "scores": [
# #         -0.7322837710380554,
# #         0.14036256074905396,
# #         -0.518636167049408
# #     ],

# # Get the average score per item
# gpt4_scores = {}
# for item in gpt4_results:
#     key = item['question']
#     scores = np.array(item['scores'])
#     # normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
#     gpt4_scores[key] = np.mean(scores)

# print(gpt4_scores)

# # Get the total average
# gpt4_total_average = np.mean(list(gpt4_scores.values()))
# print(gpt4_total_average)


import json
with open('prompt_output_dataset_bleurt.json', 'r') as f:
    gpt4_results = json.load(f)

    # iterate through each item and print the question and the distractors
    for item in gpt4_results:
        print(f"Question: {item['question']}\nCorrect Answer: {item['best_answer']}\nDistractors: {', '.join(item['distractors'])}")
        print()
