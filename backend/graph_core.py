# /backend/graph_core.py

import os
import sqlite3
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver   # ✅ corrected import

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()


# 1. Define the Graph State
class AgentState(TypedDict):
    messages: Annotated[list[HumanMessage | AIMessage], add_messages]


# 2. Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# 3. Persistent SQLite connection
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)


# 4. Core Agent Node
def call_llm(state: AgentState):
    messages = state["messages"]

    if not messages or not isinstance(messages[-1], HumanMessage):
        return state

    print(f"--- Calling Gemini LLM with {len(messages)} messages ---")
    response = llm.invoke(messages)

    return {"messages": [response]}


# 5. Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("llm_node", call_llm)
workflow.add_edge(START, "llm_node")
workflow.add_edge("llm_node", "llm_node")   # ✅ loop fixed

# Compile with memory
graph_app = workflow.compile(checkpointer=memory)

print("✅ LangGraph Agent Compiled Successfully with Gemini and Persistent SQLite Memory.")
