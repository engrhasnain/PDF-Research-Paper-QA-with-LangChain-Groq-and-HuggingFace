import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")


groq_api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(groq_api_key=groq_api_key, 
               model_name="gemma2-9b-it")

prompt = ChatPromptTemplate.from_messages([
    """
    Answer the Question based on the context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
])

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_docs, 
            st.session_state.embeddings
        )
    
user_prompt = st.text_input("Enter your question:")
if st.button("Document Embeddings"):
    create_vector_embeddings()
    st.write("Vector Database is ready for querying.")

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retreiver = st.session_state.vectors.as_retriever()
    chain = create_retrieval_chain(
        retriever=retreiver, 
        combine_docs_chain=document_chain
    )

    response = chain.invoke({"input": user_prompt})
    st.write(response['answer'])

        