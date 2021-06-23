import numpy as np
import pandas as pd
import torch
import random
import tensorflow as tf
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

def predict(sentence: str):
    processed_sentencen = clean(sentence)
    processed_sentence = removeNum(processed_sentencen)
    sent_id = tokenizer.batch_encode_plus(processed_sentence, padding=True, max_length=64,truncation=True,return_token_type_ids=False)
    ids = torch.tensor(sent_id['input_ids'])
    mask = torch.tensor(sent_id['attention_mask'])

    pred = model(ids, mask)
    _, prediction = torch.max(pred, dim=1)
    return prediction

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased', do_lower_case=True)
    bert = AutoModel.from_pretrained('dbmdz/bert-base-turkish-cased')
    model = BERT_Arch(bert)
    model.load_state_dict(torch.load("saved_weights.pt"))

    headline = "Başkandan şok açıklama! İşte o karar..."
    headline2 = "Fenerbahçe şampiyon oldu"
    prediction = predict(headline)
    print(prediction)

    #res = "Clickbait" if prediction >= 0.4 else "Non-Clickbait"
    #print(res)



