from typing import TypedDict, Dict, Optional, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain.llms.base import LLM
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
import os
from google import generativeai as genai

load_dotenv()



class AgentState(TypedDict):
    message: List[HumanMessage]


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.environ["GEMINI_API_KEY"]
)


def process(state: AgentState) -> AgentState:
    user_message = "\n".join([msg.content for msg in state['message']])
    response_text = llm.invoke(user_message)

    print("[LLM Response]", response_text.content)
    return state



graph = StateGraph(AgentState)
graph.add_node('process', process)
graph.add_edge(START, 'process')
graph.add_edge('process', END)
agent = graph.compile()


user_input = input('Enter: ')

while user_input != 'exit':
    agent.invoke({'message':[HumanMessage(content=user_input)]})
    user_input = input('Enter: ')
    