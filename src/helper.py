from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


# ✅ Extract Data From the PDF File (Generic Loader)
def load_pdf_file(data: str) -> List[Document]:
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader  # Forces it to use PyPDFLoader
    )
    documents = loader.load()
    return documents


# ✅ Filter unnecessary metadata from documents
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs


# ✅ Split long documents into chunks
def text_split(extracted_data: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# ✅ Load HF sentence transformer embeddings (384-dim)
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings


# ✅ Main document loader used in app.py
def load_pdf_documents() -> List[Document]:
    loader = DirectoryLoader(
        'C:/Users/prath/Documents/Medi Chatboat/Medical Chatboat/Medical-Chatboat/data', glob="**/*.pdf",

        loader_cls=PyPDFLoader  # ensures no unstructured dependency is needed
    )
    docs = loader.load()
    return docs
