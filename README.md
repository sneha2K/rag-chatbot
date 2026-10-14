**RAG Chatbot Using LangChain, FAISS, Ollama, and Streamlit**

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using information retrieved from a PDF document. The application uses local embeddings, a FAISS vector database, and a locally running Llama 3.2 model through Ollama. A Streamlit interface provides an interactive way to ask questions.

**Project overview**

Rather than relying only on a language model's general knowledge, this project retrieves relevant passages from a PDF and provides them as context to the model. The sample document used during development was the research paper Attention Is All You Need.

**Key features**

- Load and extract text from a PDF document.
- Split long text into overlapping chunks.
- Convert chunks into vector embeddings using sentence-transformers/all-MiniLM-L6-v2.
- Store and search embeddings with FAISS.
- Generate answers with the locally running llama3.2:3b model via Ollama.
- Ask questions through a Streamlit interface.
- Run without a paid LLM API for the local Ollama workflow.

**Architecture**

PDF document
    |
    v
PyPDFLoader (extract pages)
    |
    v
RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200)
    |
    v
HuggingFaceEmbeddings (all-MiniLM-L6-v2)
    |
    v
FAISS vector store
    ^
    | similarity search
User question --> Query embedding
    |
    v
Relevant document chunks
    |
    v
Prompt + retrieved context --> Ollama (llama3.2:3b)
    |
    v
Answer shown in Streamlit

**Tech stack**

| Component | Technology |
| --- | --- |
| Language | Python |
| PDF loader | LangChain Community `PyPDFLoader` |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector database | FAISS |
| Local language model | Llama 3.2 3B through Ollama |
| User interface | Streamlit |

**Project structure**

rag-chatbot/
├── documents/
│   └── attention_is_all_you_need.pdf
├── app.py
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md

**How I built the project**

**1. Create the project and Python environment**

mkdir rag-chatbot
cd rag-chatbot
python3 -m venv venv
source venv/bin/activate

**2. Install dependencies**

Install the project's declared dependencies:
pip install -r requirements.txt
The implementation uses LangChain, FAISS, Sentence Transformers, Streamlit, and the Ollama integration. If setting up without the requirements file, the core packages are:
pip install langchain langchain-community langchain-text-splitters langchain-huggingface langchain-ollama faiss-cpu sentence-transformers streamlit pypdf

**3. Add a PDF document**

Place the research paper at:
documents/attention_is_all_you_need.pdf

**4. Load and split the PDF**

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("documents/attention_is_all_you_need.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(pages)

print("Number of pages:", len(pages))
print("Number of chunks:", len(chunks))

During development, the sample PDF produced 15 pages and 52 chunks.

**5. Generate embeddings and create a FAISS index**
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_db = FAISS.from_documents(chunks, embeddings)

FAISS helps find the PDF chunks that are semantically closest to the user's question.

**6. Set up a free local LLM**

Install Ollama and download the model:
ollama pull llama3.2:3b

Use it in Python:
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:3b", temperature=0)

Make sure Ollama is running before launching the application.

**7. Retrieve relevant context and generate an answer**

The following is a simplified example of the RAG flow:
question = "What is self-attention in a Transformer?"
retrieved_docs = vector_db.similarity_search(question, k=4)
context = "\n\n".join(doc.page_content for doc in retrieved_docs)

prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say you do not know.

Context:
{context}

Question: {question}
"""

response = llm.invoke(prompt)
print(response.content)

**8. Launch the Streamlit app**

streamlit run streamlit_app.py

**Why RAG?**

RAG combines retrieval and generation. Instead of asking a model to answer entirely from its training knowledge, the application first retrieves relevant text from a source document. The model then uses that text to produce a grounded answer. This approach can improve relevance, although responses should still be checked against the source.

