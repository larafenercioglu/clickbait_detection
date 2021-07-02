from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer
from transformers import AutoModel
import re
import random
import requests
from bs4 import BeautifulSoup
from model import BERT_Arch

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
    _, prediction = torch.max(pred, dim=1)
    return prediction.numpy()[0]

tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-128k-cased', do_lower_case=True)
bert = AutoModel.from_pretrained('dbmdz/bert-base-turkish-128k-cased')
model = BERT_Arch(bert)
model.load_state_dict(torch.load("saved_weights_punct_128k.pt"))

'''
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.load_state_dict(torch.load("saved_weights_punct_128k.pt", map_location=device))
model.to(device)
'''

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
    app.run(debug=True, host='0.0.0.0')








