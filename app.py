from flask import Flask, render_template, request
import torch
from transformers import BertTokenizerFast
from transformers import AutoModel
from transformers import BertForSequenceClassification
import re
import numpy as np
import random
import requests
from bs4 import BeautifulSoup

def clean(tweet):
    tweet = re.sub(r"@[A-Za-z0-9ğüşöçıİĞÜŞÖÇ_]+",' ',tweet)
    tweet = re.sub(r"#[A-Za-z0-9ğüşöçıİĞÜŞÖÇ]+",' ',tweet)
    tweet = re.sub(r'https?://[A-Za-z0-9ğüşöıçİĞÜŞÖÇ./]+', ' ', tweet)
    tweet = re.sub(r"[0-9]+",' ',tweet)
    tweet = re.sub(r" +", ' ', tweet)
    return tweet

def get_headline(data: list):
    random_number = random.randint(0, len(data)-1)
    return data[random_number]

def compose_headlines():
    #####   TO GET THE NEWEST HEADLINES FOR RANDOM HEADLINE    #####
    headlines = []

    url_hürriyet = "https://www.hurriyet.com.tr/"

    r1 = requests.get(url_hürriyet)
    coverpage = r1.content
    soup1 = BeautifulSoup(coverpage, "html.parser")

    coverpage_news = soup1.find_all('span', class_='news-title')
    for new in coverpage_news:
        headlines.append(new.get_text())

    url_evrensel = "https://www.evrensel.net/"

    r1 = requests.get(url_evrensel)
    coverpage = r1.content
    soup1 = BeautifulSoup(coverpage, "html.parser")

    coverpage_news = soup1.find_all('h5', class_='card-title') #dont get anything within h5??? how to avoid <a> </a> ????
    for new in coverpage_news:
        headlines.append(new.get_text())

    with open('headlines.txt', 'w', encoding = "utf-8-sig") as f:
        for headline in headlines:
            if headline == "EVRENSEL ABONE" or headline == "EVRENSEL EGE":
                continue
            f.write(headline)
            f.write('\n')

def predict(sentence: str):
    processed_sentencen = clean(sentence)
    p = []
    p.append(processed_sentencen)
    sent_id = tokenizer.batch_encode_plus(p, padding=True, max_length=64,truncation=True,return_token_type_ids=False)
    ids = torch.tensor(sent_id['input_ids'])
    mask = torch.tensor(sent_id['attention_mask'])

    pred = model(ids, mask)
    logits = pred.logits
    logits = logits.detach().cpu().numpy()
    predictions = []
    predictions.append(logits)

    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    print(flat_predictions)
    return flat_predictions[0]

tokenizer = BertTokenizerFast.from_pretrained('dbmdz/bert-base-turkish-128k-cased', do_lower_case=True)
model = BertForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-turkish-128k-cased",
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)
model.load_state_dict(torch.load("saved_weights.pt"))

# Specify a path to save to
PATH = "saved_weights.pt"

# Save
torch.save(model.state_dict(), PATH)

# Load
device = torch.device('cpu')
model.load_state_dict(torch.load(PATH, map_location=device))

app = Flask(__name__)

compose_headlines()

with open("headlines.txt", encoding = "utf-8-sig") as data:
    headlines = []
    split = data.read().split("\n")
    for comment in split:
        headlines.append(comment)

@app.route("/", methods=["POST", "GET"])
def home():

    if request.method == "POST":
        if request.form["btn_idf"] == "submit":
            sentence = request.form["sentence"]
            with open("log_file.txt", "a") as log:
                prediction = predict(sentence)
                log.write(("Non-Clickbait" if prediction == 0 else "Clickbait") + " | " + sentence + "\n\n")

            prediction = "Non-Clickbait" if prediction == 0 else "Clickbait"
            return render_template("index.html", output=prediction, sentence=sentence)

        elif request.form["btn_idf"] == "random":
            h = get_headline(headlines)
            return render_template("index.html", sentence=h)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')








