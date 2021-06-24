from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer
from transformers import AutoModel
import re
import string
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
    p = []
    p.append(processed_sentence)
    sent_id = tokenizer.batch_encode_plus(p, padding=True, max_length=64,truncation=True,return_token_type_ids=False)
    ids = torch.tensor(sent_id['input_ids'])
    mask = torch.tensor(sent_id['attention_mask'])

    pred = model(ids, mask)
    _, prediction = torch.max(pred, dim=1)
    return prediction.numpy()[0]

tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased', do_lower_case=True)
bert = AutoModel.from_pretrained('dbmdz/bert-base-turkish-cased')
model = BERT_Arch(bert)
model.load_state_dict(torch.load("saved_weights.pt"))
app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        if request.form["btn_idf"] == "submit":
            sentence = request.form["sentence"]
            try:
                with open("log_file.txt", "a") as log:
                    prediction = predict(sentence)
                    log.write(("Non-Clickbait" if prediction == 0 else "Clickbait") + " | " + sentence + "\n\n")

            except Exception as e:
                global model
                model = BERT_Arch(bert)
                model.load_state_dict(torch.load("saved_weights.pt"))
                with open("log_file.txt", "a") as log:
                    prediction = predict(sentence)
                    log.write(("Non-Clickbait" if prediction == 0 else "Clickbait") + " | " + sentence + "\n\n")

            prediction = "Non-Clickbait" if prediction == 0 else "Clickbait"
            return render_template("index.html", output=prediction, sentence=sentence)

    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False, threaded=True, port=9000, host="0.0.0.0")








