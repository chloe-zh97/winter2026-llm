import streamlit as st
from openai import OpenAI
from agents import (
    Obnoxious_Agent,
    Query_Agent,
    Relevant_Documents_Agent,
    Answering_Agent,
    Context_Rewriter_Agent,
    Head_Agent
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

# init Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "machine-learning-textbook"
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY
)

# Initialize Agents
obnoxious_agent = Obnoxious_Agent(openai_client)
context_rewriter = Context_Rewriter_Agent(openai_client)
relevant_docs_agent = Relevant_Documents_Agent(openai_client)
answering_agent = Answering_Agent(openai_client)
query_agent = Query_Agent(pinecone_index, openai_client, embeddings)
head_agent = Head_Agent(
    openai_key=OPENAI_API_KEY,
    pinecone_key=PINECONE_API_KEY,
    pinecone_index_name=PINECONE_INDEX_NAME
)

# Streamlit UI
st.title("Multi-Agent Chatbot Demo")
agent_option = st.sidebar.selectbox(
    "Select Agent to Demo",
    [
        "Obnoxious_Agent",
        "Context_Rewriter_Agent",
        "Query_Agent",
        "Relevant_Documents_Agent",
        "Answering_Agent",
        "Head_Agent"
    ]
)

# Sample Test Cases
test_cases = {
    "Obnoxious_Agent": [
        "You are stupid!",
        "Explain supervised learning.",
        "I hate this system.",
        "What is a neural network?",
    ],
    "Context_Rewriter_Agent": [
        "Explain it.",  # assumes previous context
        "What about that?",
        "How does it work?",
        "Tell me more.",
    ],
    "Query_Agent": [
        "What is supervised learning?",
        "Explain neural networks.",
        "Who won the NBA championship in 2020?",
        "What is overfitting?",
    ],
    "Relevant_Documents_Agent": [
        "What is supervised learning?",
        "Who won the NBA championship in 2020?",
        "Explain neural networks.",
        "I hate this system.",
    ],
    "Answering_Agent": [
        "What is supervised learning?",
        "Explain overfitting.",
        "What is a neural network?",
        "How to prevent overfitting?",
    ],
    "Head_Agent": [
        "What is supervised learning?",
        "Explain neural networks.",
        "Who won the NBA championship in 2020?",
        "You are useless.",
    ],
}

# Show Test Case Selector
st.sidebar.subheader("Select Test Case")
selected_case_index = st.sidebar.selectbox(
    "Test Case",
    list(range(len(test_cases[agent_option]))),
    format_func=lambda x: test_cases[agent_option][x]
)

user_input = test_cases[agent_option][selected_case_index]

st.write(f"### Test Input for {agent_option}")
st.info(user_input)

# Run Agent Logic
if st.button("Run Demo"):

    if agent_option == "Obnoxious_Agent":
        result = obnoxious_agent.check_query(user_input)
        st.success("Obnoxious? ✅ Yes" if result else "Obnoxious? ✅ No")

    elif agent_option == "Context_Rewriter_Agent":
        # simulate conversation history
        conv_history = [
            "User: Explain supervised learning.",
            "Bot: Supervised learning is training on labeled data."
        ]
        rewritten = context_rewriter.rephrase(conv_history, user_input)
        st.write("Rewritten Query:")
        st.success(rewritten)

    elif agent_option == "Query_Agent":
        docs = query_agent.query_vector_store(user_input)
        st.write(f"Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs):
            st.write(f"**Doc {i+1}:** {doc.page_content[:300]}...")

    elif agent_option == "Relevant_Documents_Agent":
        # simulate retrieved docs
        sample_docs = [
            "Overfitting occurs when a model learns noise instead of signal.",
            "Regularization helps prevent overfitting.",
        ]
        conversation = {"query": user_input, "documents": sample_docs}
        result = relevant_docs_agent.get_relevance(conversation)
        st.write(f"Documents relevant? **{result}**")
        for i, doc in enumerate(sample_docs):
            st.write(f"**Doc {i+1}:** {doc}")

    elif agent_option == "Answering_Agent":
        sample_docs = [
            "Supervised learning is when a model is trained on labeled data.",
            "Neural networks are models inspired by the human brain.",
        ]
        conv_history = []
        answer = answering_agent.generate_response(user_input, sample_docs, conv_history)
        st.write("Answer Generated:")
        st.success(answer)

    elif agent_option == "Head_Agent":
        st.write("Running Head Agent loop (CLI simulation):")
        # Run a single query through head agent pipeline
        rewritten_query = head_agent.context_rewriter.rephrase(
            head_agent.conversation_history, user_input
        )

        if head_agent.obnoxious_agent.check_query(rewritten_query):
            response = "Your query is inappropriate."
        else:
            retrieved_docs = head_agent.query_agent.query_vector_store(rewritten_query)
            docs_text = [doc.page_content for doc in retrieved_docs]
            relevance = head_agent.relevant_docs_agent.get_relevance({
                "query": rewritten_query,
                "documents": docs_text
            })
            # if relevance == "No":
            #     response = "Your question is outside the scope of this knowledge base."
            # else:
            #     response = head_agent.answering_agent.generate_response(
            #         rewritten_query, docs_text, head_agent.conversation_history
            #     )
            response = head_agent.answering_agent.generate_response(
                    rewritten_query, docs_text, head_agent.conversation_history
                )
        st.success(response)