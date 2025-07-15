import os
import requests
from dotenv import load_dotenv
import trafilatura
from bs4 import BeautifulSoup
import requests
from playwright.sync_api import sync_playwright
import json


load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

class SerperSearch:
    def __init__(self):
        self.api_key = SERPER_API_KEY
        self.url = "https://google.serper.dev/news"
        self.headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

    def invoke(self, query: str) -> dict:
        payload = {"q": query,"tbs": "qdr:h"}
        try:
            response = requests.post(self.url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"[SerperSearch Error] {e}")
            return {"results": []}

def serper_search():
    return SerperSearch()


def scrape_website(url):
    try:
        html = requests.get(url).text
        result = trafilatura.extract(html, with_metadata=True, output_format='json')
        if result:
            data = json.loads(result)
            return data
        return None
    except Exception as e:
        print(f'FAILED TO SCRAPE {url}: {e}')



def scrape_x_tweets(username, max_tweets):
    tweets = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # Set headless=False to debug visually
        page = browser.new_page()
        page.goto(f"https://x.com/{username}", timeout=60000)

        page.wait_for_selector("article", timeout=10000)

        articles = page.locator("article").all()
        for article in articles[:max_tweets]:
            try:
                tweets.append(article.inner_text())
            except:
                continue

        browser.close()

    tweet_text = ''
    for i,tweet in enumerate(tweets):
        tweet_text += (f"\n--- Tweet {i+1} ---\n{tweet}")

    return tweet_text




