---
title: Insurance Rag Bot
emoji: 📉
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 5.41.1
app_file: app.py
pinned: false
license: mit
short_description: insurance-rag-bot
---
 "LLM-powered insurance query system using"

📄 File Upload (PDF/DOCX)

📚 FAISS vector store

🤖 Gemini API with 1M token context

🔍 Token chunking + metadata filtering

🧠 mpnet-based semantic embeddings

💬 Retrieval-Augmented Generation (RAG)

🖥️ Gradio interactive UI

💰 Token usage + cost tracking
---

# 🧠 Gemini + FAISS Insurance Policy Query Analyzer

This project is an intelligent **Insurance Policy Query Analyzer** built using **Gradio**, **Google Gemini**, **FAISS**, and **LangChain**. It allows users to upload insurance policy documents and ask questions about them. The system provides accurate, grounded answers using Retrieval-Augmented Generation (RAG).

---

## 🔧 Features

* Upload and index insurance documents (PDF, DOCX).
* Chunking and embedding using HuggingFace + LangChain.
* FAISS vector search for fast document retrieval.
* Gemini (via `google.generativeai`) for answer generation.
* Gradio-based user interface for quick interaction.
* Optimized for faster processing and cleaner outputs.

---

## 🧱 Internal Architecture

```plaintext
        +--------------------+
        |   User Interface   | (Gradio)
        +--------------------+
                  |
                  v
        +---------------------+
        |  Document Uploader  |
        +---------------------+
                  |
        +-----------------------------+
        | Document Loader & Splitter |  -->  LangChain loaders
        +-----------------------------+
                  |
        +------------------------+
        | Text Chunk Embeddings |  --> HuggingFaceEmbeddings
        +------------------------+
                  |
        +----------------+
        |  FAISS Vector  |
        |   Store (RAG)  |
        +----------------+
                  |
        +-------------------------+
        | Query + Retrieved Chunks|
        +-------------------------+
                  |
        +------------------+
        | Gemini (LLM API) |
        +------------------+
                  |
        +----------------+
        |  Final Answer  |
        +----------------+
```

---

## 🧠 Thought Process Behind Design

### 🔍 Why FAISS + RAG?

Insurance policies are lengthy and complex. We needed:

* **Fast document retrieval** ➝ FAISS is fast, scalable, and integrates well with LangChain.
* **Contextual answers** ➝ Using Gemini ensures the language model has up-to-date, deep understanding.
* **Chunking** ➝ Token-level chunking ensures each vector has complete and coherent information.

### 🌐 Why Gemini?

Gemini is:

* Multimodal (future extensibility).
* Strong at long-context understanding.
* Simple to configure via `google.generativeai`.

### 🧱 Modular Design

The code is broken into:

1. **Document Upload + Indexing**
2. **Embedding + Vector Store Management**
3. **User Query + Retrieval**
4. **LLM Answer Generation**

This makes it easy to maintain, extend, or replace components (e.g., swap Gemini with GPT-4, or FAISS with ChromaDB).

### 🖼️ Why Gradio?

* Fast to prototype.
* Clean UI with drag-and-drop file upload.
* Easily deployable on Hugging Face Spaces or locally.

---

## ⚙️ Tech Stack

| Tool           | Purpose                               |
| -------------- | ------------------------------------- |
| Gradio         | User interface                        |
| Gemini API     | LLM for natural language generation   |
| FAISS          | Vector similarity search              |
| HuggingFace    | Text embeddings (`all-MiniLM-L6-v2`)  |
| LangChain      | Document loaders, chunking, retrieval |
| PyPDFLoader    | PDF parsing                           |
| Docx2txtLoader | DOCX parsing                          |

---

## 🚀 Future Improvements

* Streamlit or React UI alternative.
* Support for audio input or phone-based queries.
* Upload multiple policies at once.
* Store chat history with persistent user sessions.
* Add OCR for image-based policies.

---

## 📁 Folder Structure (Recommended)

```plaintext
├── app.py                  # Main Gradio app
├── utils/
│   ├── loaders.py          # PDF/DOCX loading functions
│   ├── splitter.py         # Chunking logic
│   ├── embeddings.py       # Embedding generator
│   ├── retriever.py        # FAISS vector search
│   └── llm.py              # Gemini answer generation
├── requirements.txt
├── README.md
```

---

## ✅ How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   python app.py
   ```




