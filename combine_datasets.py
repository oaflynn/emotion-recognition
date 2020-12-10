import emoji
import pandas as pd
from sklearn.model_selection import train_test_split
import helpers as h

TEST_PORTION = 0.2

df_dict = {
    'text': [],
    'label': []
}

# Process BTD
print('processing BTD')
btd = open('/Volumes/LEXAR/Emotion Recgonition Datasets/Jan9-2012-tweets-clean.txt', 'r')
btd_text = []
btd_labels = []
for line in btd:
    data = line.split('\t')
    text = data[1].strip()
    text = h.preprocessing(text)
    
    label = data[2][3:].strip()
    if label not in h.labels:
        continue

    btd_text.append(text)
    btd_labels.append(label)

# Split off BTD test set
train_text, test_text, train_labels, test_labels = train_test_split(btd_text, btd_labels, test_size=TEST_PORTION)
btd_test = {'text': test_text, 'label': test_labels}
btd_test = pd.DataFrame(btd_test)
btd_test = btd_test.sample(frac=1)
btd_test.to_csv('BTD Test.csv')

btd_text = train_text
btd_labels = train_labels

# Process CBET
print('processing CBET')
cbet = pd.read_csv('/Volumes/LEXAR/Emotion Recgonition Datasets/CBET.csv')
cbet_text = []
cbet_labels = []
for index, row in cbet.iterrows():
    text = h.preprocessing(row['text'])
    cbet_text.append(text)
    for col in cbet.columns[2:]:
        if row[col] == 1:
            cbet_labels.append(col)
            break

# Split off CBET test set
train_text, test_text, train_labels, test_labels = train_test_split(cbet_text, cbet_labels, test_size=TEST_PORTION)
cbet_test = {'text': test_text, 'label': test_labels}
cbet_test = pd.DataFrame(cbet_test)
cbet_test = cbet_test.sample(frac=1)
cbet_test.to_csv('CBET Test.csv')

cbet_text = train_text
cbet_labels = train_labels

# Process SemEval
print('processing SemEval')
emotions = ['anger', 'fear', 'joy', 'sadness']
sets = [('development', 'dev'), ('training', 'train'), ('test-gold', 'test-gold')]
# Aggregate training sets
semeval_text = []
semeval_labels = []
for dataset in sets[:1]:
    for emotion in emotions:
        file = open('/Volumes/LEXAR/Emotion Recgonition Datasets/SemEval2018-Task1-all-data/English/EI-oc/' + dataset[0] + '/2018-EI-oc-En-' + emotion + '-' + dataset[1] + '.txt', 'r')
        for line in file:
            data = line.split('\t')
            if data[1] == 'Tweet':
                continue
            if data[3][0] != '0' and data[2] in h.labels:
                text = h.preprocessing(data[1])
                semeval_text.append(text)

                semeval_labels.append(data[2])
# Aggregate test sets
semeval_test = {'text': [], 'label': []}
for emotion in emotions:
    file = open('/Volumes/LEXAR/Emotion Recgonition Datasets/SemEval2018-Task1-all-data/English/EI-oc/' + sets[2][0] + '/2018-EI-oc-En-' + emotion + '-' + sets[2][1] + '.txt', 'r')
    for line in file:
        data = line.split('\t')
        if data[1] == 'Tweet':
            continue
        if data[3][0] != '0' and data[2] in h.labels:
            text = h.preprocessing(data[1])
            semeval_test['text'].append(text)
            semeval_test['label'].append(data[2])

semeval_test = pd.DataFrame(semeval_test)
semeval_test = semeval_test.sample(frac=1)
semeval_test.to_csv('SemEval Test.csv')

# Combine training samples
train_text = btd_text + cbet_text + semeval_text
train_labels = btd_labels + cbet_labels + semeval_labels

# Save train set
train_set = {'text': train_text, 'label': train_labels}
train_set = pd.DataFrame(train_set)
train_set = train_set.sample(frac=1)
train_set.to_csv('Training set.csv')