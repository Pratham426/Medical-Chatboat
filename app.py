from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings, load_pdf_documents
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Pinecone v3 client
from pinecone import Pinecone as PineconeClient, ServerlessSpec
# Pinecone integration for LangChain
from langchain_community.vectorstores import Pinecone as LangchainPinecone

# Flask app initialization
app = Flask(__name__)

# Load .env variables
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Step 1: Load embeddings
embeddings = download_hugging_face_embeddings()

# Step 2: Define Pinecone index name
index_name = "medical-chatbot"

# Step 3: Initialize Pinecone client
pc = PineconeClient(api_key=PINECONE_API_KEY)

# Step 4: Create index if not exists
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # embedding dimension for all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Step 5: Load documents
documents = load_pdf_documents()

# Step 6: Create LangChain Pinecone vectorstore
docsearch = LangchainPinecone.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=index_name
)

# Step 7: Create retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Step 8: Setup Groq LLM
chatModel = ChatGroq(model="llama3-8b-8192")

# Step 9: Define system prompt
system_prompt = (
    "You are a Medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Step 10: Create prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Step 11: Create RAG chain
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Step 12: Routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    return str(response["answer"])

# Step 13: Run app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
