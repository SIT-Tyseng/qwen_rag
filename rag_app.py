# rag_app.py

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
import torch
import gradio as gr
from typing import Optional, List

# --- Load Documents ---
def load_documents():
    # loader = DirectoryLoader("docs/", glob="*.txt", loader_cls=TextLoader)
    loader = DirectoryLoader("docs/", glob="*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)

# --- Setup Vectorstore ---
def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    return Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")

# --- Load Local Qwen Model ---
def load_qwen_model(model_path="Qwen/Qwen2-7B"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).eval()
    return tokenizer, model

# --- Custom LLM Wrapper ---
class LocalQwenLLM(LLM):
    tokenizer: object
    model: object

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def _llm_type(self) -> str:
        return "qwen"

# --- Build QA Chain ---
def build_qa_chain(vectorstore, tokenizer, model):
    llm = LocalQwenLLM(tokenizer=tokenizer, model=model)
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

    print("Loading Qwen model...")
    tokenizer, model = load_qwen_model()

    print("Building QA chain...")
    qa_chain = build_qa_chain(vectorstore, tokenizer, model)

    print("Starting Gradio UI...")
    interface = gr.Interface(
        fn=answer_question,
        inputs="text",
        outputs=[
            gr.Textbox(label="Answer"),
            gr.JSON(label="Sources")
        ],
        title="🧠 Qwen RAG Chatbot (Local)",
        description="Ask questions based on your local documents using Qwen."
    )
    interface.launch(server_name="0.0.0.0", server_port=7860)