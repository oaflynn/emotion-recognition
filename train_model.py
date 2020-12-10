import pandas as pd
import helpers as h
from sklearn.model_selection import train_test_split
from transformers import XLNetTokenizer, XLNetForSequenceClassification, Trainer, TrainingArguments, AdamW
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# USER INPUT CONSTANTS
VALIDATION_PORTION = .1
MAX_ENCODING_LENGTH = 64
BATCH_SIZE = 16

# Load data sets
training_df = pd.read_csv('Training set.csv')
training_df = pd.concat([training_df[training_df['label']=='joy'], training_df[training_df['label']=='anger']], axis=0) # Only include joy and anger samples for now
training_df = training_df[:10000] # Only use a portaion of training set
training_df = training_df.sample(frac=1) # Shuffle

cbet_df = pd.read_csv('CBET Test.csv')
cbet_df = pd.concat([cbet_df[cbet_df['label']=='joy'], cbet_df[cbet_df['label']=='anger']], axis=0)

# Only using cbet so far
btd_df = pd.read_csv('BTD Test.csv')
semeval_df = pd.read_csv('SemEval Test.csv')

# Split sample dfs
train_texts, train_labels = h.sample_split(training_df)
cbet_texts, cbet_labels = h.sample_split(cbet_df)
btd_texts, btd_labels = h.sample_split(btd_df)
semeval_texts, semeval_labels = h.sample_split(semeval_df)

# Split training set into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=VALIDATION_PORTION)
tran_texts = train_texts[:-7]
train_labels = train_labels[:-7]

# There are a few nan's in here... not sure why, but I will just change them to strings for now because
# I don't want to misalign the labels
train_texts = [str(s) for s in train_texts]
val_texts = [str(s) for s in val_texts]
cbet_texts = [str(s) for s in cbet_texts]
btd_texts = [str(s) for s in btd_texts]
semeval_texts = [str(s) for s in semeval_texts]

# Tokenize texts
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_ENCODING_LENGTH)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_ENCODING_LENGTH)
cbet_encodings = tokenizer(cbet_texts, truncation=True, padding=True, max_length=MAX_ENCODING_LENGTH, return_tensors="pt")
btd_encodings = tokenizer(btd_texts, truncation=True, padding=True, max_length=MAX_ENCODING_LENGTH)
semeval_encodings = tokenizer(semeval_texts, truncation=True, padding=True, max_length=MAX_ENCODING_LENGTH)

# Torch dataset class
class EmotionsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = EmotionsDataset(train_encodings, train_labels)
val_dataset = EmotionsDataset(val_encodings, val_labels)
cbet_dataset = EmotionsDataset(cbet_encodings, cbet_labels)
btd_dataset = EmotionsDataset(btd_encodings, btd_labels)
semeval_dataset = EmotionsDataset(semeval_encodings, semeval_labels)

# Instantiate model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2, mem_len=1024)
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in tqdm(range(3)):
    for i, batch in tqdm(enumerate(train_loader)):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        if i % 50 == 0:
             print('iteration ', i, ' loss: ', loss)
        loss.backward()
        optim.step()

model.save_pretrained("./model")

# Evaluate
model.eval()
print('cbet')
h.eval_with_dataset(cbet_dataset, model, device, BATCH_SIZE)
#print('val')
#h.eval_with_dataset(val_dataset, model, device, BATCH_SIZE)
#print('training')
#h.eval_with_dataset(train_dataset, model, device, BATCH_SIZE)

#h.eval_with_dataset(btd_dataset, model, device, BATCH_SIZE)
#h.eval_with_dataset(semeval_dataset, model, device, BATCH_SIZE)