from openai import OpenAI
from dotenv import load_dotenv
import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
open_ai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
# print(open_ai_api_key)
# print(pinecone_api_key)

# Python
class Obnoxious_Agent:
    def __init__(self, client) -> None:
        # TODO: Initialize the client and prompt for the Obnoxious_Agent
        self.client = client

        # Default strict prompt
        self.system_prompt = (
            "You are a content moderation agent. "
            "Your task is to determine whether a user's query is obnoxious, "
            "offensive, toxic, abusive, hateful, or inappropriate.\n\n"
            "If the query is obnoxious, respond ONLY with: Yes\n"
            "If the query is not obnoxious, respond ONLY with: No\n"
            "Do not include any explanation."
        )

    def set_prompt(self, prompt):
        # TODO: Set the prompt for the Obnoxious_Agent
        self.system_prompt = prompt

    def extract_action(self, response) -> bool:
        # TODO: Extract the action from the response
        answer = response.strip().lower()
        if "yes" in answer:
            return True
        elif "no" in answer:
            return False
        else:
            return False

    def check_query(self, query):
        # TODO: Check if the query is obnoxious or not
        completion = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0  # deterministic output
        )
        response_text = completion.choices[0].message.content
        return self.extract_action(response_text)

class Context_Rewriter_Agent:
    def __init__(self, openai_client):
        # TODO: Initialize the Context_Rewriter agent
        self.client = openai_client
        self.system_prompt = (
            "You are a query rewriting agent.\n"
            "Your task is to rewrite the user's latest question into a clear, "
            "fully self-contained question.\n\n"
            "Rules:\n"
            "- Resolve pronouns and ambiguous references using the conversation history.\n"
            "- Keep the original meaning.\n"
            "- Do NOT answer the question.\n"
            "- Only return the rewritten question.\n"
        )

    def rephrase(self, user_history, latest_query):
        # TODO: Resolve ambiguities in the final prompt for multiturn situations
        # Convert history into readable text block
        history_text = ""
        for i, msg in enumerate(user_history):
            history_text += f"Turn {i+1}: {msg}\n"

        user_prompt = (
            f"Conversation History:\n{history_text}\n"
            f"Latest User Question:\n{latest_query}\n\n"
            "Rewrite the latest question into a standalone, unambiguous question."
        )

        completion = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        rewritten_query = completion.choices[0].message.content.strip()
        return rewritten_query

class Query_Agent:
    def __init__(self, pinecone_index, openai_client, embeddings) -> None:
        # TODO: Initialize the Query_Agent agent
        self.index = pinecone_index
        self.client = openai_client
        self.embeddings = embeddings
        self.namespace = "ns2500"

        self.system_prompt = (
            "You are a domain relevance classifier.\n"
            "Given a user query and retrieved documents, determine whether "
            "the documents are relevant to answering the query.\n\n"
            "Respond ONLY with:\n"
            "Yes - if the documents are relevant\n"
            "No - if the documents are not relevant\n"
            "Do not provide explanations."
        )

    def query_vector_store(self, query, k=5):
        # TODO: Query the Pinecone vector store
        vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            namespace=self.namespace
        )

        results = vectorstore.similarity_search(query, k=k)
        return results

    def set_prompt(self, prompt):
        # TODO: Set the prompt for the Query_Agent agent
        self.system_prompt = prompt

    def extract_action(self, response, query = None):
        # TODO: Extract the action from the response
        decision = response.strip().lower()
        if "yes" in decision:
            return True
        elif "no" in decision:
            return False
        else:
            # fallback safety
            return False

    def check_relevance(self, query, k=5):
        """
        Full pipeline:
        - Retrieve documents
        - Check relevance
        - Return structured result
        """
        documents = self.query_vector_store(query, k=k)

        if not documents:
            return {
                "is_relevant": False,
                "documents": [],
                "raw_matches": []
            }

        # Step 2: Extract text from LangChain Document objects
        docs_text_list = [doc.page_content for doc in documents]
        docs_text = "\n\n".join(docs_text_list)

        # Step 3: Build relevance checking prompt
        user_prompt = (
            f"User Query:\n{query}\n\n"
            f"Retrieved Documents:\n{docs_text}\n\n"
            "Are these documents relevant to answering the user's query?"
        )

        completion = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )

        response_text = completion.choices[0].message.content
        is_relevant = self.extract_action(response_text, query=query)

        return {
            "is_relevant": is_relevant,
            "documents": docs_text_list,   # plain text list
            "raw_matches": documents       # original Document objects
        }

class Answering_Agent:
    def __init__(self, openai_client) -> None:
        # TODO: Initialize the Answering_Agent
        self.client = openai_client
        self.system_prompt = (
            "You are a helpful AI assistant.\n"
            "Answer the user's question using ONLY the provided documents.\n\n"
            "Rules:\n"
            "- Use only the provided documents as knowledge.\n"
            "- If the answer is not in the documents, say:\n"
            "  'I don't have enough information in the provided documents.'\n"
            "- Be clear and concise.\n"
        )

    def generate_response(self, query, docs, conv_history, k=5):
        # TODO: Generate a response to the user's query
        if not docs:
            return "I don't have enough information in the provided documents."

        # Combine top-k documents
        selected_docs = docs[:k]
        docs_text = "\n\n".join(selected_docs)

        # Format conversation history
        history_text = ""
        for i, msg in enumerate(conv_history):
            history_text += f"Turn {i+1}: {msg}\n"

        user_prompt = (
            f"Conversation History:\n{history_text}\n\n"
            f"User Question:\n{query}\n\n"
            f"Relevant Documents:\n{docs_text}\n\n"
            "Answer the question using the provided documents."
        )

        completion = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )

        response = completion.choices[0].message.content.strip()
        return response

class Relevant_Documents_Agent:
    def __init__(self, openai_client) -> None:
        # TODO: Initialize the Relevant_Documents_Agent
        self.client = openai_client
        self.system_prompt = (
            "You are a relevance checking agent.\n"
            "Your job is to determine whether the retrieved documents "
            "are relevant to answering the user's query.\n\n"
            "Respond ONLY with:\n"
            "Yes - if the documents are relevant\n"
            "No - if they are not relevant\n"
            "Do not provide explanation."
        )

    def get_relevance(self, conversation) -> str:
        # TODO: Get if the returned documents are relevant
        """
        conversation should contain:
        {
            'query': str,
            'documents': list[str]
        }
        """
        query = conversation.get("query", "")
        documents = conversation.get("documents", [])

        if not documents:
            return "No"

        docs_text = "\n\n".join(documents)

        user_prompt = (
            f"User Query:\n{query}\n\n"
            f"Retrieved Documents:\n{docs_text}\n\n"
            "Are these documents relevant to the user's query?"
        )

        completion = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )

        response = completion.choices[0].message.content.strip()

        # Enforce strict Yes/No output
        if "yes" in response.lower():
            return "Yes"
        elif "no" in response.lower():
            return "No"
        else:
            return "No"

class Head_Agent:
    def __init__(self, openai_key, pinecone_key, pinecone_index_name) -> None:
        # TODO: Initialize the Head_Agent
        self.openai_client = OpenAI(api_key=openai_key)

        self.pc = Pinecone(api_key=pinecone_key)
        self.index = self.pc.Index(pinecone_index_name)

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key
        )

        self.conversation_history = []

        # Setup sub-agents
        self.setup_sub_agents()

    def setup_sub_agents(self):
        # TODO: Setup the sub-agents
        self.obnoxious_agent = Obnoxious_Agent(self.openai_client)
        self.context_rewriter = Context_Rewriter_Agent(self.openai_client)
        self.query_agent = Query_Agent(
            pinecone_index=self.index,
            openai_client=self.openai_client,
            embeddings=self.embeddings
        )
        self.relevant_docs_agent = Relevant_Documents_Agent(self.openai_client)
        self.answering_agent = Answering_Agent(self.openai_client)

    def main_loop(self):
        # TODO: Run the main loop for the chatbot
        print("Multi-Agent Chatbot Started (type 'exit' to quit)\n")
        while True:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            # 1. Rewrite context (multi-turn support)
            rewritten_query = self.context_rewriter.rephrase(
                self.conversation_history,
                user_input
            )

            # 2. Check obnoxious content
            if self.obnoxious_agent.check_query(rewritten_query):
                response = "Your query is inappropriate. Please ask something else."
                print("Bot:", response)
                continue

            # 3. Retrieve documents
            retrieved_docs = self.query_agent.query_vector_store(rewritten_query)
            docs_text = [doc.page_content for doc in retrieved_docs]

            # 4. Check document relevance
            relevance = self.relevant_docs_agent.get_relevance({
                "query": rewritten_query,
                "documents": docs_text
            })

            if relevance == "No":
                response = "Your question is outside the scope of this knowledge base."
                print("Bot:", response)
                continue

            # 5. Generate final answer
            response = self.answering_agent.generate_response(
                query=rewritten_query,
                docs=docs_text,
                conv_history=self.conversation_history
            )

            # 6. Update conversation history
            self.conversation_history.append(f"User: {user_input}")
            self.conversation_history.append(f"Bot: {response}")
            print("Bot:", response)
