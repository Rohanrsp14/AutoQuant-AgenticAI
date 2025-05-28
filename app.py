import streamlit as st
from agent import ask_agent

st.set_page_config(page_title="AutoQuant LLM Agent", layout="centered")

st.title("🤖 AutoQuant: Your Financial Research Agent")

query = st.text_input("Ask a market question:", placeholder="e.g., What is the sentiment around Nvidia stock today?")

if query:
    with st.spinner("Thinking..."):
        response = ask_agent(query)
        st.markdown(response, unsafe_allow_html=True)  # ✅ this renders the image properly
