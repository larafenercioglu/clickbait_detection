import torch
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
import re
import numpy as np

def clean(tweet):
    tweet = re.sub(r"@[A-Za-z0-9ğüşöçıİĞÜŞÖÇ_]+",' ',tweet)
    tweet = re.sub(r"#[A-Za-z0-9ğüşöçıİĞÜŞÖÇ]+",' ',tweet)
    tweet = re.sub(r'https?://[A-Za-z0-9ğüşöıçİĞÜŞÖÇ./]+', ' ', tweet)
    tweet = re.sub(r" +", ' ', tweet)
    return tweet

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

if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained('dbmdz/bert-base-turkish-128k-cased', do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(
        "dbmdz/bert-base-turkish-128k-cased",
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False,
    )
    model.load_state_dict(torch.load("saved_weights.pt"))

    with open("headlines.txt", encoding = "utf-8-sig") as data:
        headlines = []
        split = data.read().split("\n")
        for comment in split:
            headlines.append(comment)

    predictions = []

    for headline in headlines:
        prediction = predict(headline)
        predictions.append(prediction)

    preds = {}
    for i in range(len(headlines)):
        preds[headlines[i]] = "Clickbait" if predictions[i] == 1 else "Non-Clickbait"

    print(preds)





