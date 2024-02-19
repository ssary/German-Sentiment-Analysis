from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import random
import numpy as np
from sklearn.metrics import classification_report

LR = 2e-5
EPOCHS = 2
BATCH_SIZE = 16
MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment" # use this to finetune the sentiment classifier
MAX_TRAINING_EXAMPLES = 200000 # set this to -1 if you want to use the whole training set (869376 rows)

# loading the German sentiment dataset
files = """test_labels.txt
test_text.txt
train_labels.txt
train_text.txt
val_labels.txt
val_text.txt""".split('\n')

# Changing the structure
dataset_dict = {}
for i in ['train','val','test']:
  dataset_dict[i] = {}
  for j in ['text','labels']:
    dataset_dict[i][j] = open(f"/home/cluster_home/s0591103/workspace/HPC_fine/{i}_{j}.txt", encoding='latin-1').read().rstrip('\n').split('\n')
    if j == 'labels':
      dataset_dict[i][j] = [int(x) for x in dataset_dict[i][j]]

if MAX_TRAINING_EXAMPLES > 0:
    combined_list = list(zip(dataset_dict['train']['text'], dataset_dict['train']['labels']))
    random.shuffle(combined_list)
    shuffled_texts, shuffled_labels = zip(*combined_list)
    dataset_dict['train']['text'] = list(shuffled_texts)[:MAX_TRAINING_EXAMPLES]
    dataset_dict['train']['labels'] = list(shuffled_labels)[:MAX_TRAINING_EXAMPLES]
# Init tokenizer of the model
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

# Running the tokenizer
train_encodings = tokenizer(dataset_dict['train']['text'], max_length=128, truncation=True, padding=True)
val_encodings = tokenizer(dataset_dict['val']['text'], max_length=128, truncation=True, padding=True)
test_encodings = tokenizer(dataset_dict['test']['text'], max_length=128, truncation=True, padding=True)

# Defining the dataset, adding encodings
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MyDataset(train_encodings, dataset_dict['train']['labels'])
val_dataset = MyDataset(val_encodings, dataset_dict['val']['labels'])
test_dataset = MyDataset(test_encodings, dataset_dict['test']['labels'])

# Fine tuning settings/parameters
training_args = TrainingArguments(
    output_dir='./results_all',                   # output directory
    num_train_epochs=EPOCHS,                  # total number of training epochs
    per_device_train_batch_size=BATCH_SIZE,   # batch size per device during training
    per_device_eval_batch_size=BATCH_SIZE,    # batch size for evaluation
    warmup_steps=100,                         # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                        # strength of weight decay
    logging_dir='./logs',                     # directory for storing logs
    logging_steps=10,                         # when to print log
    load_best_model_at_end=True,              # load or not best model at the end
    evaluation_strategy='steps'
)

num_labels = len(set(dataset_dict["train"]["labels"]))
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)

# Train
trainer = Trainer(
    model=model,                              # the pretrained Transformers model to be trained
    args=training_args,                       # training arguments, defined above
    train_dataset=train_dataset,              # training dataset
    eval_dataset=val_dataset                  # evaluation dataset
)

trainer.train()

# Save the model
trainer.save_model("./results_all/best_model")

# Evaluation on the test set
test_preds_raw, test_labels , _ = trainer.predict(test_dataset)
test_preds = np.argmax(test_preds_raw, axis=-1)

# Classification report including accuracy.
print(classification_report(test_labels, test_preds, digits=3))