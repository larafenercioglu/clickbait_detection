# importing the necessary packages
import requests
from bs4 import BeautifulSoup

headlines = []

################################################################
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
################################################################

with open('headlines.txt', 'w', encoding = "utf-8-sig") as f:
    for headline in headlines:
        if headline == "EVRENSEL ABONE" or headline == "EVRENSEL EGE":
            continue
        f.write(headline)
        f.write('\n')