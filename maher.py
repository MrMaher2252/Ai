import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Professional AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Professional AI Assistant")
st.caption("Powered by Groq • Built with Streamlit")

# -----------------------------
# API key check
# -----------------------------
# Check if GROQ_API_KEY environment variable is set
if "GROQ_API_KEY" not in os.environ:
    st.error("GROQ_API_KEY environment variable is not set.")
    st.stop()

# Get the API key from environment variable
api_key = os.environ.get("GROQ_API_KEY")

# -----------------------------
# LLM initialization
# -----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.6,
    max_tokens=500,
)

# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Display chat history
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# User input
# -----------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # Save and show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Convert history to LangChain messages
    lc_messages = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for chunk in llm.stream(lc_messages):
            full_response += chunk.content
            placeholder.markdown(full_response)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )