from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
import requests, os, base64
from sentence_transformers import SentenceTransformer, util
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from newsapi import NewsApiClient
from dotenv import load_dotenv

class AgentState(TypedDict):
    user_search_request: str
    query_type: str
    search_queries: List[str]
    summarize_queries: List[dict]
    final_report: List[dict]
    prompt: str
    img_data: List[dict]

load_dotenv()

NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
FREEPIK_API_KEY = os.environ.get('FREEPIK_API_KEY')
TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY')


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI( model="gemini-2.0-flash", google_api_key=os.environ["GEMINI_API_KEY"])
search_tool = TavilySearch(max_results=3,topic="news",search_depth="advanced")
newsapi = NewsApiClient(api_key=NEWS_API_KEY)


def classify_query(state: AgentState) -> AgentState:
    prompt = f"""
    Classify the following query into one of the following types:
    - factual
    - trending_news
    - opinion
    - comparison
    - technical

    Only return the type with no punctuation or explanation.

    Query: {state['user_search_request']}
    """
    response = llm.invoke(prompt)
    state['query_type'] = response.content.strip().lower()
    return state

def split_topic(state: AgentState) -> AgentState:
    topic = state['user_search_request']
    split_prompt = f"""
    Break the following topic into 3 to 5 distinct sub-questions. One question per line, no numbering.
    Topic: {topic}
    """
    result = llm.invoke(split_prompt)
    subquestions = [line.strip() for line in result.content.splitlines() if line.strip()]
    state['search_queries'] = subquestions
    return state

def split_topic_or_not(state: AgentState) -> str:
    if state['query_type'] in ['factual', 'technical', 'comparison']:
        return "split"
    else:
        state['search_queries'] = [state['user_search_request']]
        return "no_split"

def fetch_web_and_news_results(query: str, query_type: str) -> List[dict]:
    results = []
    try:
        web = search_tool.invoke(query)
        if web.get("results"):
            results.extend(web["results"])
    except Exception as e:
        print(f"Search tool failed: {e}")

    
    if query_type == "trending_news":
        try:
            news = newsapi.get_everything(q=query, page_size=5, sort_by="publishedAt")
            for article in news.get("articles", []):
                results.append({
                    "url": article.get("url"),
                    "content": article.get("description") or article.get("content"),
                    "source": article.get("source", {}).get("name", "NewsAPI")
                })
        except Exception as e:
            print(f"NewsAPI error: {e}")

    return results


def deduplicate_sources(sources: List[dict], threshold=0.85) -> List[dict]:
    unique = []
    embeddings = []
    for src in sources:
        text = src.get("content", "")
        if not text:
            continue
        emb = embedding_model.encode(text, convert_to_tensor=True)
        if all(util.cos_sim(emb, e).item() < threshold for e in embeddings):
            embeddings.append(emb)
            unique.append(src)
    return unique

def search_node(state: AgentState) -> AgentState:
    responses = []
    for query in state['search_queries']:
        results = fetch_web_and_news_results(query, state['query_type'])
        deduped = deduplicate_sources(results)
        responses.append({"query": query, "results": deduped})
    state['summarize_queries'] = responses
    return state

def summarize_node(state: AgentState) -> AgentState:
    summary = []
    for item in state['summarize_queries']:
        question = item['query']
        sources = item.get('results', [])
        if not sources:
            continue
        source_text = "\n".join(
            f"[{i+1}] {s['content']} ({s['url']})" for i, s in enumerate(sources)
        )
        sum_prompt = f"""
        Based on the following sources, write a clear and informative answer to the question.
        Use inline citations like [1], [2] where appropriate.

        Question: {question}

        Sources:
        {source_text}

        Answer:
        """
        response = llm.invoke(sum_prompt)
        summary.append({"question": question, "summary": response.content.strip()})
    state['final_report'] = summary
    return state

def img_prompter(state: AgentState) -> AgentState:
    context = "\n\n".join([s["summary"] for s in state["final_report"]])
    prompt = f"""
    Generate a cinematic, cartoon-style image prompt based on this content:
    {context}
    """
    response = llm.invoke(prompt)
    state["prompt"] = response.content.strip()
    return state

def img_generator(state: AgentState) -> AgentState:
    url = "https://api.freepik.com/v1/ai/text-to-image"
    req = {
        "prompt": state['prompt'],
        "seed": 42,
        "num_images": 1,
        "image": {"size": "square_1_1"},
    }
    headers = {
        "x-freepik-api-key": FREEPIK_API_KEY,
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=req, headers=headers)
    if response.status_code == 200:
        data = response.json()
        base64_image = data.get("data", [{}])[0].get("base64")
        if base64_image:
            image_path = "generated_image.jpg"
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(base64_image))
            state['img_data'] = [{"file_path": os.path.abspath(image_path), "status": "saved"}]
        else:
            state['img_data'] = [{"status": "no_base64_returned"}]
    else:
        state['img_data'] = [{"status": "error", "code": response.status_code, "message": response.text}]
    return state


graph = StateGraph(AgentState)

graph.add_node("classify_query", classify_query)
graph.add_node("split_topic", split_topic)
graph.add_node("search_node", search_node)
graph.add_node("summarize_node", summarize_node)
graph.add_node("img_prompter", img_prompter)
graph.add_node("img_generator", img_generator)

graph.add_edge(START, "classify_query")
graph.add_conditional_edges("classify_query", split_topic_or_not, {
    "split": "split_topic",
    "no_split": "search_node"
})

graph.add_edge("split_topic", "search_node")
graph.add_edge("search_node", "summarize_node")
graph.add_edge("summarize_node", "img_prompter")
graph.add_edge("img_prompter", "img_generator")
graph.add_edge("img_generator", END)


compiled_graph = graph.compile()


initial_state = {
    "user_search_request": "Elon Musk latest Tweet",
    "query_type": "",
    "search_queries": [],
    "summarize_queries": [],
    "final_report": [],
    "prompt": "",
    "img_data": []
}

result = compiled_graph.invoke(initial_state)
print(result["final_report"])
print(result.get("img_data"))