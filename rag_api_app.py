import os
import requests
from typing import Optional, List

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
import gradio as gr

# --- Load Documents ---
def load_documents():
    loader = DirectoryLoader("docs/", glob="*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)

# --- Setup Vectorstore ---
def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    return Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")

# --- Qwen3 API-based LLM Wrapper ---
class QwenAPILLM(LLM):
    api_key: str = ""
    model_name: str = "qwen3"
    max_tokens: int = 512

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "false"
        }

        data = {
            "model": self.model_name,
            "input": {
                "prompt": prompt
            },
            "parameters": {
                "max_output_tokens": self.max_tokens
            }
        }

        response = requests.post(
            "https://api.dashscope.cn/api/v1/services/aigc/text-generation/generation ",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            return response.json()['output']['text']
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

    @property
    def _llm_type(self) -> str:
        return "qwen3-api"

# --- Build QA Chain ---
def build_qa_chain(vectorstore, api_key: str):
    llm = QwenAPILLM(api_key=api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# --- Gradio Interface Function ---
def answer_question(question):
    result = qa_chain(question)
    return result["result"], result["source_documents"]

# --- Main Execution ---
if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()

    print("Setting up vector store...")
    vectorstore = setup_vectorstore(docs)

    # Set your DashScope API key here or use environment variable
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "YOUR_API_KEY_HERE")

    print("Building QA chain...")
    qa_chain = build_qa_chain(vectorstore, DASHSCOPE_API_KEY)

    print("Starting Gradio UI...")
    interface = gr.Interface(
        fn=answer_question,
        inputs="text",
        outputs=[
            gr.Textbox(label="Answer"),
            gr.JSON(label="Sources")
        ],
        title="🧠 Qwen3 RAG Chatbot (via API)",
        description="Ask questions based on your PDF documents using Qwen3 via DashScope API."
    )
    interface.launch(server_name="0.0.0.0", server_port=7860)