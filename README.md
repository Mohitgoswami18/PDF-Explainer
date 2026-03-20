# 📄 RAG PDF Chat Application

A simple **Retrieval-Augmented Generation (RAG)** application that allows users to upload a PDF and ask questions based on its content.

---

## 🚀 Features

* 📄 Upload any PDF document
* 🔍 Ask questions related to the document
* 🧠 Uses RAG (Retrieval-Augmented Generation)
* ⚡ Fast similarity search using FAISS
* 🤖 Powered by Hugging Face LLM

---

## 🧠 How It Works

1. Upload a PDF
2. The document is split into smaller chunks
3. Each chunk is converted into embeddings
4. Embeddings are stored in a FAISS vector database
5. User query is matched with relevant chunks
6. LLM generates an answer based on retrieved context

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Framework:** LangChain
* **Vector Store:** FAISS
* **Embeddings:** Hugging Face (sentence-transformers/all-MiniLM-L6-v2)
* **LLM:** Hugging Face (meta-llama/Llama-3.2-1B-Instruct)

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-pdf-chat.git
cd rag-pdf-chat
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add environment variables

Create a `.env` file:

```env
HUGGINGFACEHUB_API_TOKEN=your_api_key_here
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📸 Demo

Upload a PDF and ask questions like:

* "What is software engineering?"
* "Summarize this document"
* "What are the key points?"

---

## 📁 Project Structure

```
PDF-Explainer/
│── app.py
│── requirements.txt
│── .env
```

---

## ⚠️ Limitations

* Works best with text-based PDFs
* Performance depends on model and hardware
* Large PDFs may take time to process

## 🤝 Contributing

Feel free to fork this repository and improve it!

---

## ⭐ Acknowledgements

* LangChain
* Hugging Face
* Streamlit

---

## 📬 Contact

If you liked this project, feel free to connect!

