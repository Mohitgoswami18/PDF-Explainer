import streamlit as st
import tempfile
from dotenv import load_dotenv

# Required Libraries

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

load_dotenv()

st.set_page_config(page_title="RAG PDF Chat", page_icon="📄")
st.title("📄 Chat with your PDF (RAG)")
st.write("Upload a PDF and ask questions!")

# Upload PDF

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# Query input

query = st.text_input("Ask a question about the PDF")

if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Document Loader
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(docs)

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store
    vector_store = FAISS.from_documents(chunks, embedding_model)

    # Retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    def get_relevant_docs(raw_relevant_docs):
        return "\n\n".join([doc.page_content for doc in raw_relevant_docs])

    # Prompt
    prompt = PromptTemplate(
        template="""I am giving you a question and related context for it and you have to answer the question based on the context and if the question is out of context just say sorry the context is not related to this question.
    ```

    Question: {query}

    Context: {context}""",
    input_variables=["query", "context"]
    )

    # LLM
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-1B-Instruct",
        task="text-generation"
    )

    model = ChatHuggingFace(llm=llm)

    parser = StrOutputParser()

    # Chain
    parallel_chain = RunnableParallel({
        "query": RunnablePassthrough(),
        "context": retriever | RunnableLambda(get_relevant_docs)
    })

    final_chain = parallel_chain | prompt | model | parser

    # Run when query is entered
    if query:
        with st.spinner("Thinking... 🤔"):
            output = final_chain.invoke(query)

        st.subheader("💡 Answer:")
        st.write(output)