import os
from dotenv import load_dotenv
from uuid import uuid4
import streamlit as st

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# ---- ENV ----
load_dotenv()
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'QNA_Chatbot'
os.environ['HUGGING_FACE_TOKEN'] = os.getenv('HUGGING_FACE_TOKEN')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


# ---- PDF FUNCTIONS ----
def load_pdf(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def text_splitter(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=60)
    return splitter.split_text(text)


def create_vectorDB(text):
    embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vector = Chroma.from_texts(text, embedding)
    return vector


def output_formatter(docs):
    return "\n\n".join(d.page_content for d in docs)


# ---- MODEL + PROMPT ----
model = ChatGroq(model="llama-3.3-70b-versatile")

prompt_template = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers questions using the PDF context.

Use the PDF context as the main source of truth.
If the answer is partially available, use what is available.
If the answer is not mentioned, say: "I don't know based on the PDF."
Use the recent conversation history to understand follow-ups.
Keep the answer precise and short.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ---- SESSION STATE ----
if "loaded" not in st.session_state:
    st.session_state.loaded = False
if "store" not in st.session_state:
    st.session_state.store = {}
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())


# ---- MEMORY ----
def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]


# ---- UI ----
st.title("📄 PDF Q&A Chatbot")

col1, col2 = st.columns([1, 2])

# ---- LEFT SIDE ----
with col1:
    st.subheader("Upload PDF")
    pdf = st.file_uploader("Drag & Drop or Browse", type=["pdf"])

    if pdf:
        pdf_text = load_pdf(pdf)
        chunks = text_splitter(pdf_text)
        vector = create_vectorDB(chunks)
        retriever = vector.as_retriever()
        st.session_state.retriever = retriever

        def get_context(x):
            return output_formatter(retriever.invoke(x["question"]))

        base_chain = (
            RunnablePassthrough.assign(context=get_context)
            | prompt_template
            | model
        )

        st.session_state.rag_chain = RunnableWithMessageHistory(
            base_chain,
            get_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

        st.session_state.loaded = True
        st.success("PDF loaded successfully!")

    else:
        st.info("Upload a PDF to start chatting.")


# ---- RIGHT SIDE ----
with col2:
    st.subheader("Chat")

    if st.session_state.loaded:
        history = get_history(st.session_state.session_id).messages

        # Display stored history from RunnableWithMessageHistory
        for msg in history:
            if isinstance(msg, HumanMessage):
                st.markdown(f"**User:** {msg.content}")
            else:
                st.markdown(f"**Bot:** {msg.content}")

        # chat input
        user_msg = st.chat_input("Ask something from your PDF")

        if user_msg:
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke(
                    {"question": user_msg},
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )

            # directly display bot response without rerun
            st.markdown(f"**User:** {user_msg}")
            st.markdown(f"**Bot:** {response.content}")

    else:
        st.info("Upload a PDF to enable chat.")


