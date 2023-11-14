# Dataset
The truthful_qa dataset was split into train (60%), validation (20%) and test dataset (20%)
```python
import pickle    
with open('train_dataset.pickle', 'rb') as output:
    train_dataset = pickle.load(output)
with open('val_dataset.pickle', 'rb') as output:
    val_dataset = pickle.load(output)
with open('test_dataset.pickle', 'rb') as output:
    test_dataset = pickle.load(output)
```