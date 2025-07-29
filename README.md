# PDF Research Papers Q&A with LangChain Groq API and HuggingFace Embeddings

This project is a **Streamlit-based** application that enables users to query PDF research papers using **LangChain**, **Groq API** (with the `gemma2-9b-it` model), and **HuggingFace Embeddings**. It uses **FAISS** for vector search and a retrieval-based pipeline to provide accurate, context-driven answers.  

---

## 🚀 Features

- 📄 Load multiple research papers (PDF)  
- ✂️ Split documents into chunks for better retrieval  
- 🧠 Generate vector embeddings using HuggingFace  
- 🔍 Search documents using FAISS vector store  
- 🤖 Answer questions using Groq’s `gemma2-9b-it` model  
- 🎨 User-friendly interface powered by Streamlit  

---

## 🧱 Tech Stack

- [Streamlit](https://streamlit.io/)  
- [LangChain](https://www.langchain.com/)  
- [Groq API](https://console.groq.com/)  
- [HuggingFace Embeddings](https://huggingface.co/)  
- [FAISS](https://github.com/facebookresearch/faiss)  
- Python `3.10+`  

---

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/groq-langchain-pdf-qa.git
cd groq-langchain-pdf-qa
```
### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```
If using Anaconda/Conda then
```bash
conda create -m venv
conda activate venv
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup .env file
```bash
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```
### 5. Add PDF Files
Place your PDF files inside the ```research_papers/ ``` folder.


### 🧠 How It Works
LangChain Retrieval Flow
This app uses a retrieval-augmented pipeline:

``` bash
User Question → Vector Search (FAISS) → Relevant Chunks → Groq LLM → Response
```

- Document Loader: Loads PDFs using PyPDFDirectoryLoader.
- Text Splitter: Splits documents into chunks for better indexing.
- Embeddings: Generates vector embeddings using HuggingFace.
- FAISS Vector Store: Stores and retrieves document chunks.
- LLM: Groq’s gemma2-9b-it provides precise, context-aware answers.

  
### ▶️ Run the App
```bash
streamlit run app.py
```
### 🧪 Usage Steps
- Start the app using the command above.
- Click "Document Embeddings" to process and index all PDFs.
- Enter your question in the text input.
- Get accurate answers based on your research papers!

### 🌟 Star This Repo
If you found this useful, please give it a ⭐️ to support the project!
