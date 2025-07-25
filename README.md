# Financial Advisor Agent

A conversational **Financial Advisor Chatbot** powered by LLMs and LangGraph’s REACT framework — capable of reasoning, using tools, and giving smart financial answers.

---

## Live Demo
[Click here to use the Financial Advisor Agent](https://your-demo-link.com)  

---

## What Can This Agent Do?

This intelligent assistant can:

- **Retrieve stock and company information**  
  → e.g., “What’s the share price of Tesla?”

- **Convert currencies using live data**  
  → e.g., “Convert 1000 USD to Japanese Yen”

- **Perform financial math calculations**  
  → e.g., “If I invest $5000 for 4 years at 7% interest, what’s my return?”

- **Explain financial concepts and comparisons**  
  → e.g., “What’s Apple’s P/E ratio and how does it compare to Microsoft?”

All actions are handled through **LLM reasoning**, tool selection, and chained execution using LangGraph’s REACT-style architecture.

---

## Tools Used

The agent leverages the following tools integrated with the LangGraph framework:

| Tool Name              | Description                                                |
|------------------------|------------------------------------------------------------|
| **Google Finance Tool**    | Retrieves stock prices and company metadata              |
| **Alpha Vantage Tool**     | Fetches real-time currency and forex exchange data       |
| **Python REPL Tool**       | Performs mathematical and financial computations         |

---

## Architecture Overview

This is a **REACT-style agent** built using [LangGraph](https://langchain-ai.github.io/langgraph/):

- **LLM Reasoning Node (`reason`)**  
  Selects the appropriate tool and thinks through user queries

- **Tool Execution Node (`run_tool`)**  
  Executes selected tools dynamically based on user needs

- **Summarization Node (`summarize`)**  
  Combines and summarizes tool results for a final user response

- **Conditional Edges**  
  Dynamically routes between reasoning and tool execution


---


## ⚙️ How to Run Locally

```bash
git clone https://github.com/Viswa-Prakash/Financial_Agent_MultiTool_ReACT.git
cd Financial_Agent_MultiTool_ReACT
pip install -r requirements.txt
python main.py
