import streamlit as st
from openai import OpenAI
from agents import (
    Obnoxious_Agent,
    Query_Agent,
    Relevant_Documents_Agent,
    Answering_Agent,
    Context_Rewriter_Agent,
)
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# load env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
print(OPENAI_API_KEY)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "machine-learning-textbook"

# Initialize Agents
obnoxious_agent = Obnoxious_Agent(openai_client)
context_rewriter = Context_Rewriter_Agent(openai_client)
relevant_docs_agent = Relevant_Documents_Agent(openai_client)
answering_agent = Answering_Agent(openai_client)

# set up Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY
)

query_agent = Query_Agent(
    pinecone_index=pinecone_index,
    openai_client=openai_client,
    embeddings=embeddings,
)

# Streamlit UI
st.title("Multi-Agent Chatbot Demo")
st.sidebar.title("Select Agent to Demo")

agent_option = st.sidebar.selectbox(
    "Choose Agent",
    ["Obnoxious Agent", "Query Agent", "Relevant Docs Agent", "Answering Agent"]
)

# Sample Test Cases
test_cases = {
    "Obnoxious Agent": [
        "You are stupid!",
        "Explain supervised learning.",
        "I hate this system.",
        "What is a neural network?",
    ],
    "Query Agent": [
        "What is supervised learning?",
        "Explain neural networks.",
        "Who won the NBA championship in 2020?",  # irrelevant
        "What is overfitting?",
    ],
    "Relevant Docs Agent": [
        "What is supervised learning?",
        "Who won the NBA championship in 2020?",
        "Explain neural networks.",
        "I hate this system.",
    ],
    "Answering Agent": [
        "What is supervised learning?",
        "Explain overfitting.",
        "What is a neural network?",
        "How to prevent overfitting?",
    ],
}

# Show Test Case Selector
st.sidebar.subheader("Select Test Case")
selected_case_index = st.sidebar.selectbox(
    "Test Case", list(range(len(test_cases[agent_option]))),
    format_func=lambda x: test_cases[agent_option][x]
)

user_input = test_cases[agent_option][selected_case_index]

st.write(f"### Test Input for {agent_option}")
st.info(user_input)

# Run Agent Logic
if st.button("Run Demo"):

    if agent_option == "Obnoxious Agent":
        result = obnoxious_agent.check_query(user_input)
        if result:
            st.error("Obnoxious? ✅ Yes")
        else:
            st.success("Obnoxious? ✅ No")

    elif agent_option == "Query Agent":
        docs = query_agent.query_vector_store(user_input)
        st.write(f"Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs):
            st.write(f"**Doc {i+1}:** {doc.page_content[:300]}...")

    elif agent_option == "Relevant Docs Agent":
        # For demo, we simulate retrieved docs
        sample_docs = [
            "Overfitting occurs when a model learns noise instead of signal.",
            "Regularization helps prevent overfitting.",
        ]
        conversation = {"query": user_input, "documents": sample_docs}
        result = relevant_docs_agent.get_relevance(conversation)
        st.write(f"Documents relevant? **{result}**")
        st.write("Sample Documents:")
        for i, doc in enumerate(sample_docs):
            st.write(f"**Doc {i+1}:** {doc}")

    elif agent_option == "Answering Agent":
        # For demo, simulate retrieved docs
        sample_docs = [
            "Supervised learning is when a model is trained on labeled data.",
            "Neural networks are models inspired by the human brain.",
        ]
        conv_history = []
        answer = answering_agent.generate_response(user_input, sample_docs, conv_history)
        st.write("Answer Generated:")
        st.success(answer)
