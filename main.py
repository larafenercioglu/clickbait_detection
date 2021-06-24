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

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased', do_lower_case=True)
    bert = AutoModel.from_pretrained('dbmdz/bert-base-turkish-cased')
    model = BERT_Arch(bert)
    model.load_state_dict(torch.load("saved_weights.pt"))

    headline = "Başkandan şok açıklama! İşte o karar..."
    headline2 = "Fenerbahçe şampiyon oldu"
    headline3 = "CHP bugün oylamada erken seçim olacağının belirtti"
    headline4 = "Corona aşısı Mayıs ayı itibariyle tüm vatandaşlara uygulanacak"
    headline5 = "İşte o futbolcu transfer oldu"
    headline6 = "Tartışma yaratan karar: İşveren çalışanını aşı olmaya zorlayabilir mi?"
    headlines = []
    headlines.append(headline)
    headlines.append(headline2)
    headlines.append(headline3)
    headlines.append(headline4)
    headlines.append(headline5)
    headlines.append(headline6)

    predictions = []
    prediction = predict(headline)
    prediction2 = predict(headline2)
    prediction3 = predict(headline3)
    prediction4 = predict(headline4)
    prediction5 = predict(headline5)
    prediction6 = predict(headline6)

    predictions.append(prediction)
    predictions.append(prediction2)
    predictions.append(prediction3)
    predictions.append(prediction4)
    predictions.append(prediction5)
    predictions.append(prediction6)

    preds = {}
    for i in range(len(headlines)):
        preds[headlines[i]] = "Clickbait" if predictions[i] == 1 else "Non-Clickbait"

    print(preds)





