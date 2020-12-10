import nltk
import sys
import re
import emoji
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score

tokenizer = nltk.TweetTokenizer()
stopwords = nltk.corpus.stopwords.words('english')
negative_contractions = {
    "aren't": "are not",
    "isn't": "is not",
    "can't": "can not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "shouldn't": "should not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "wouldn't": "would not"
}
emotion_labels = ['joy', 'sadness', 'surprise', 'fear', 'anger', 'love', 'disgust', 'guilt', 'thankfulness']

def label_to_onehot(label_list, label):
    onehot = [0 for x in range(len(label_list))]
    onehot[label_list.index(label)] = 1
    return onehot

def onehot_to_label(label_list, onehot):
    one_index = onehot.index(1)
    return label_list[one_index]

def label_to_int(label, label_list=emotion_labels):
    return emotion_labels.index(label)

def int_to_label(int_label, label_list=emotion_labels):
    return emotion_labels[int_label]

def preprocessing(text):
    text = text.lower() # Lowercase
    # Expand neagtive contractions
    for contraction in negative_contractions.keys():
        text = text.replace(contraction, negative_contractions[contraction], sys.maxsize)
    text = re.sub('!!+', '! <repeat>', text) # Replace repeating !
    text = re.sub('\?\?+', '? <repeat>', text) # Replace repeating ?
    # TO DO: Replace slang words
    tokens = tokenizer.tokenize(text) # Tokenize
    tokens = [emoji.demojize(t) for t in tokens] # Descriptive details for emojis
    tokens = [t for t in tokens if t not in stopwords]# Remove stopwords
    tokens = ['<number>' if t.isnumeric() else t for t in tokens] # Add number tokens
    tokens = [t for t in tokens if t != ''] # Somewhere, a lot of empty string tokens get added.. take them out
    tokens = [t.replace('#', '') if t[0] == '#' else t for t in tokens] # Strip hashtags

    return ' '.join(tokens)

def sample_split(df):
    texts = []
    labels = []
    for i, row in df.iterrows():
        texts.append(row['text'])
        labels.append(emotion_labels.index(row['label']))
    return texts, labels

#http://mccormickml.com/2019/07/22/BERT-fine-tuning/#52-evaluate-on-test-set
def eval_with_dataset(dataset, model, device, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size)
    
    predictions = []
    true_labels = []
    
    # Predict 
    for i, batch in enumerate(data_loader):
        if i % 100 == 0:
            print(str(i) + '/' + str(len(data_loader)) + ' batches')
        
        # Add batch to GPU
        batch = tuple(batch[t].to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_token_type, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
    
    print('Done predicting')
    result_correctness = []
    for i in range(len(true_labels)):
        for j in range(len(true_labels[i])):
            pred_labels_i = np.argmax(predictions[i][j]).flatten()[0]
            true_labels_i = true_labels[i][j]
            result_correctness.append(1 if pred_labels_i == true_labels_i else 0)
    
    correct = sum(result_correctness)
    total = len(result_correctness)
    accuracy = correct / total
    print(str(correct) + ' correct out of ' + str(total) + ': ' + str(accuracy) + '%')

onehots = {label:label_to_onehot(emotion_labels, label) for label in emotion_labels}
def add_onehots(df):
    df[emotion_labels] = 0
    for label in emotion_labels:
        df.loc[df['label'] == label, emotion_labels] = onehots[label]