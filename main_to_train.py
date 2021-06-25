import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
from transformers import BertTokenizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re
from sklearn.metrics import accuracy_score
import string
import time
import datetime

from model import BERT_Arch

# specify GPU
device = torch.device("cuda")

import nltk
nltk.download('stopwords')
nltk.download('punkt')

def clean(tweet):
    tweet = re.sub(r"@[A-Za-z0-9ğüşöçıİĞÜŞÖÇ_]+",' ',tweet)
    tweet = re.sub(r"#[A-Za-z0-9ğüşöçıİĞÜŞÖÇ]+",' ',tweet)
    tweet = re.sub(r'https?://[A-Za-z0-9ğüşöıçİĞÜŞÖÇ./]+', ' ', tweet)
    tweet = re.sub(r" +", ' ', tweet)
    exclude = set(string.punctuation)
    tweet = ''.join(ch for ch in tweet if ch not in exclude)
    return tweet

def removeNum(tweet):
    tweet = re.sub(r"[0-9]+",' ',tweet)
    return tweet

df = pd.read_csv('tr_clickbait_dataset.csv', encoding="utf-8-sig")
df.dropna(inplace=True)

# check class distribution
print(df['clickbait'].value_counts(normalize = True))

data_clean = []
for headline in df['headline']:
    headlined = str(headline)
    headlined = re.sub(r"Diken",' ',headlined)
    cleaned_headline = clean(headlined)
    cleanedd_headline = removeNum(cleaned_headline)
    h = re.sub(r"…",' ', cleanedd_headline)
    data_clean.append(h)

x = pd.Series(data_clean)
x = x.tolist()
df['tr_headline'] = x
del df['headline']
df.reset_index(drop=True, inplace=True)

def plotMostCommons(df):
    #for clickabait sentiments
    df_clickbait=df[df["clickbait"]==1]

    #for only unigrams
    token_list=[]
    exclude = set(string.punctuation)
    exclude.add('’')
    stop_words=stopwords.words("turkish")

    for i,r in df_clickbait.iterrows():
        text=''.join(ch for ch in df_clickbait["tr_headline"][i] if ch not in exclude) #remove punctuations from the text in order not to distort frequencies
        #text=''.join(ch for ch in df_clickbait["headline"][i]) #with punctuation
        tokens=word_tokenize(text)
        tokens=[tok.lower() for tok in tokens if tok not in stop_words] #remove stopwords from the text in order not to distort frequencies
        token_list.extend(tokens)

    frequencies=Counter(token_list)
    frequencies_sorted=sorted(frequencies.items(), key=lambda k: k[1],reverse=True)
    top_15=dict(frequencies_sorted[0:15])

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # Example data
    ngram = top_15.keys()
    y_pos = np.arange(len(ngram))
    performance = top_15.values()


    ax.barh(y_pos, performance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ngram)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Counts')
    ax.set_title('Top-15 Most Common Unigrams in Clickbait Headlines')

    plt.show()

    #for negative sentiments
    df_nonclickbait=df[df["clickbait"]==0]
    exclude.add('‘')
    #for only unigrams
    token_list=[]

    for i,r in df_nonclickbait.iterrows():
        text=''.join(ch for ch in df_nonclickbait["tr_headline"][i] if ch not in exclude) #remove punctuations from the text in order not to distort frequencies
        tokens=word_tokenize(text)
        tokens=[tok.lower() for tok in tokens if tok not in stop_words] #remove stopwords from the text in order not to distort frequencies
        token_list.extend(tokens)

    frequencies=Counter(token_list)
    frequencies_sorted=sorted(frequencies.items(), key=lambda k: k[1],reverse=True)
    top_15=dict(frequencies_sorted[0:15])

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # Example data
    ngram = top_15.keys()
    y_pos = np.arange(len(ngram))
    performance = top_15.values()

    ax.barh(y_pos, performance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ngram)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Counts')
    ax.set_title('Top-15 Most Common Unigrams in Non Clickbait Headlines')

    plt.show()

plotMostCommons(df)

train_text, temp_text, train_labels, temp_labels = train_test_split(df['tr_headline'], df['clickbait'],
                                                                    random_state=42,
                                                                    test_size=0.3,
                                                                    stratify=df['clickbait'])

# we will use temp_text and temp_labels to create validation and test set
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=42,
                                                                test_size=0.5,
                                                                stratify=temp_labels)

# import BERT-base pretrained model
bert = AutoModel.from_pretrained('dbmdz/bert-base-turkish-cased')

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('dbmdz/bert-base-turkish-cased')

def plot_sentence_embeddings_length(text_list, tokenizer):
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t), text_list))
    tokenized_texts_len = list(map(lambda t: len(t), tokenized_texts))
    fig, ax = plt.subplots(figsize=(8, 5));
    ax.hist(tokenized_texts_len, bins=40);
    ax.set_xlabel("Length of Comment Embeddings");
    ax.set_ylabel("Number of Comments");
    plt.show()
    return

plot_sentence_embeddings_length(train_text.values,tokenizer)

"""Padding all the samples to the maximum length is not efficient: it’s better to pad the samples when we’re building a batch, as then we only need to pad to the maximum length in that batch, and not the maximum length in the entire dataset. This can save a lot of time and processing power when the inputs have very variable lengths!"""

max_seq_len = 64 #her batch için bunu güncellesennnnnn

# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = max_seq_len,
    padding=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = max_seq_len,
    padding=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = max_seq_len,
    padding=True,
    truncation=True,
    return_token_type_ids=False
)

# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

# for validation set
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

# for test set
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#define a batch size
batch_size = 32 #CAN BE CHANGED????

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = SequentialSampler(val_data), batch_size=batch_size)

# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = False

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)

# optimizer from hugging face transformers
from transformers import AdamW #CAN BE CHANGED????????????

# define the optimizer
optimizer = AdamW(model.parameters(), lr = 5e-5)

from sklearn.utils.class_weight import compute_class_weight

#compute the class weights
class_wts = compute_class_weight('balanced', np.unique(train_labels), train_labels)

print(class_wts)

# convert class weights to tensor
weights= torch.tensor(class_wts,dtype=torch.float)
weights = weights.to(device)

# loss function #CAN BE CHANGED????????????
cross_entropy  = nn.NLLLoss(weight=weights)

# number of training epochs
epochs = 20 #CAN BE CHANGED?????????????

# function to train the model
def train():

    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds=[]

    # iterate over batches
    for step,batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    #returns the loss and predictions
    return avg_loss, total_preds

# function for evaluating the model
def evaluate():

    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step,batch in enumerate(val_dataloader):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:

            # Calculate elapsed time in minutes.
            #elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds,labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

#for each epoch
for epoch in range(epochs):

    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

    #train model
    train_loss, _ = train()

    #evaluate model
    valid_loss, _ = evaluate()

    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')

    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

#load weights of best model
path = 'saved_weights.pt'
model.load_state_dict(torch.load(path))

# get predictions for test data
with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()

# model's performance
preds = np.argmax(preds, axis = 1)
print(classification_report(test_y, preds))

# confusion matrix
print(pd.crosstab(test_y, preds))