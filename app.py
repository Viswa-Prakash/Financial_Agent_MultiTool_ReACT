import streamlit as st
from langchain_core.messages import HumanMessage
from ReAct_Agent import financial_agent  # make sure your agent's code is in ReAct_Agent.py

st.set_page_config(page_title="Financial Advisor Agent", page_icon="💸")
st.title("Financial Advisor ReAct Agent")
st.markdown(
    """
Ask financial questions like:
- `"What's the share price of Tesla and convert 100 shares to Japanese Yen?"`
- `"What's Apple's P/E ratio and how does it compare with Microsoft?"`
- `"If I invest $5000 for 4 years at 7% annual return, what will I have?"`

You’ll receive a **single, final answer**, neatly formatted with advice.
"""
)

with st.form("financial_query_form"):
    user_query = st.text_area("Your financial or investment question:", height=80)
    submitted = st.form_submit_button("Ask Agent")

if submitted and user_query.strip():
    with st.spinner("Analyzing your question..."):
        response = financial_agent.invoke({
            "messages": [HumanMessage(content=user_query)]
        })
        # Only display the very last agent message in the answer, formatted appropriately
        agent_messages = [m for m in response['messages'] if getattr(m, "role", "").lower() in ("assistant", "ai", "system") or hasattr(m, "content")]
        if agent_messages:
            final_answer = agent_messages[-1].content.strip()
            # Put the last message in a highlighted box per your example
            st.markdown(
                f"""
**Here’s a clear summary of your requests and answers:**  

{final_answer}
""",
                unsafe_allow_html=True
            )
        else:
            st.warning("Sorry, I wasn't able to generate an answer. Please try rephrasing your question.")

st.markdown("---")
st.caption("Powered by LangGraph, LangChain, and Streamlit.")