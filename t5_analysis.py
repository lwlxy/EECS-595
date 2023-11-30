# Remember, to run visually, you can use: tensorboard --logdir ./results/runs

# Read each file that ends with -metrics.txt from /results folder
# and calculate the average of the normalized scores
import os
import glob
import numpy as np

path = './results'
files = glob.glob(os.path.join(path, '*-metrics.txt'))
# save as dictionary - the prefix of each metrics file should be the key
# example filename: ep_5_lr_0.0002_wd_0.1-metrics.txt
metrics_scores = {}
for file in files:
    key = file.split('\\')[-1]
    with open(file, 'r') as f:
        # convert file to json format
        curr_metrics = eval(f.read())
        # normalize test_scores from 0-1
        if('eval_scores' in curr_metrics): 
            scores = curr_metrics['eval_scores']
            loss = curr_metrics['eval_loss']
        else: 
            scores = curr_metrics['test_scores']
            loss = curr_metrics['test_loss']
        scores = np.array(scores)
        normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        # get average of normalized scores and loss as dict
        metrics_scores[key] = {'average_scores': np.mean(scores), 'normalized_scores': np.mean(normalized_scores), 'loss': loss}

# now write metrics_scores to an easy to read file
# one line per key, with the average score, normalized score and loss
with open('hyperparameter_analysis.txt', 'w') as f:
    for key, value in metrics_scores.items():
        f.write('%s:%s\n' % (key, value))

print(metrics_scores)


