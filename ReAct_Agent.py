import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper
from langchain_community.utilities import AlphaVantageAPIWrapper
from langchain_experimental.tools.python.tool import PythonREPLTool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_community.tools import tool

# ------------- .env & API keys
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
alphavantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")

# ------------- LLM
llm = init_chat_model("openai:gpt-4.1")

# ------------- Tools
google_finance_tool = GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper())
alpha_vantage_tool = AlphaVantageAPIWrapper(alphavantage_api_key=alphavantage_api_key)

@tool
def get_alpha_vantage_tool(from_currency: str, to_currency: str) -> str:
    """Retrieves the currency exchange rate between two currencies."""
    return alpha_vantage_tool.run(from_currency=from_currency, to_currency=to_currency)

repl_tool = PythonREPLTool()

tools = [google_finance_tool, get_alpha_vantage_tool, repl_tool]

# ------------- ReAct prompt
prompt = """
You are a financial advisor agent. You can:
- Use the Google Finance tool for stock/company price and info.
- Use the Alpha Vantage tool for real-time currency/forex data.
- Use the Python REPL for basic math or finance calculations (returns, averages, conversions).
ALWAYS explain your steps, choose the right tool, and combine info before you answer the user.
"""

# ------------- State
class State(TypedDict):
    messages:Annotated[list[AnyMessage], add_messages]

# ------------- Reasoning Node
def reason_and_select_tool(state:State):
    """
    LLM reasoning step that selects which tool to use based on the message history.
    """
    llm_with_tool = llm.bind_tools(tools)
    system_message = {"role":"system", "content":prompt}
    messages = [system_message] + state["messages"]
    
    response = llm_with_tool.invoke(messages)
    return {"messages": state["messages"] + [response]}

# ------------- ToolNode
tool_node = ToolNode(tools=tools)

# ------------- Final Summary Node
def summarize_final(state:State):
    """
    # LLM synthesizes all tool results and reasoning for a final response

    """
    user_prompt = state["messages"][0]
    history = state["messages"]

    system_out = {
        "role" : "system",
        "content" : (
            "Summarize the steps and results above as a human-friendly financial advisory answer. "
            "Give advice or next steps if appropriate, with clear numbers or explanations."
        )
    }

    response = llm.invoke([system_out, user_prompt] + history)
    return {"messages": [response]}


# ------------- Routing Logic
def should_continue(state:State):
    # If the last message includes a tool call, loop to tool execution again
    last = state["messages"][-1]
    return "run_tool" if getattr(last, "tool_calls", []) else "summarize"

# ------------- Build the graph!
builder = StateGraph(State)
builder.add_node("reason", reason_and_select_tool)
builder.add_node("run_tool", tool_node)
builder.add_node("summarize", summarize_final)
builder.add_edge(START, "reason")
builder.add_conditional_edges(
    "reason", should_continue, {"run_tool": "run_tool", "summarize": "summarize"}
)
builder.add_edge("run_tool", "reason")
builder.add_edge("summarize", END)
financial_agent = builder.compile()

