# Streamlit user interface which takes an PDF from the user 


# Required Libraries 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Document Loader
loader = PyPDFLoader("SE_unit1.pdf")

docs = loader.load()
# print(len(docs))
# print(docs[0].page_content)

# Creating the content in single paragraph
content = "\n\n".join([docs[i].page_content for i in range(len(docs))])
# print(content)

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

chunks = text_splitter.split_text(content)

# print(len(chunks))

# creating an embedding model
embedding_model = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(chunks, embedding_model) 
