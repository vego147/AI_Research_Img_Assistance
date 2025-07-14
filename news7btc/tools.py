import os
import requests
from dotenv import load_dotenv



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




