# Summarize_Visualize

Summarize_Visualize is an automated research pipeline that classifies a user's query, fetches up-to-date information from the web and news sources, summarizes it with citations, and generates a corresponding AI image. It is built using LangGraph, vector stores, and multiple third-party APIs.

## Features

- Query classification (factual, trending, opinion, technical, etc.)
- Web and news search using APIs like Serper.dev, Tavily, and NewsAPI
- Deduplication using semantic similarity (Sentence Transformers)
- Context-aware summarization with inline citations
- AI image prompt generation
- Image generation using Freepik AI API
- Fully orchestrated using LangGraph

## Project Structure

```
Summarize_Visualize/
│
├── news7btc/                # Main application logic
│   ├── main.py              # Workflow logic
│   ├── tools.py             # Custom tools like serper_search
│   ├── test1/2/3.ipynb      # Notebooks for testing
│   ├── prompts.py           # Prompt crafting logic
│   └── generated_image.jpg  # Output image
│
├── deep_research/           # Placeholder for future components
├── .env                     # API keys (excluded from Git)
├── .gitignore               # Ignored files and folders
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Requirements

- Python 3.10+
- API Keys:
  - `NEWS_API_KEY` (NewsAPI)
  - `FREEPIK_API_KEY` (Freepik AI)
  - `TAVILY_API_KEY` (Tavily or Serper.dev)
  - `GEMINI_API_KEY` (Google Gemini for summarization)

## Setup

1. Clone the repository:

```bash
git clone https://github.com/vego147/Summarize_Visualize.git
cd Summarize_Visualize
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file and add your API keys:

```env
NEWS_API_KEY=your_newsapi_key
FREEPIK_API_KEY=your_freepik_key
TAVILY_API_KEY=your_tavily_or_serper_key
GEMINI_API_KEY=your_google_gemini_key
```

## Example Usage

```python
initial_state = {
    "user_search_request": "Elon Musk latest tweet",
    "query_type": "",
    "search_queries": [],
    "summarize_queries": [],
    "final_report": [],
    "prompt": "",
    "img_data": []
}

result = compiled_graph.invoke(initial_state)
```

## License

This project is licensed under the MIT License.

## Future Improvements

- Add support for more search providers
- UI for summarization + image generation
- Dockerized deployment
