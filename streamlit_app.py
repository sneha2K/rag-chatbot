import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

# Load the PDF
pdf_path = "documents/attention_is_all_you_need.pdf"


# Create the local LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# Page configuration
st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📄"
)

# Title
st.title("📄 RAG Document Chatbot")

st.write("Upload a PDF and ask questions about the document.")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    # Save the uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_pdf_path = temp_file.name

    # Read the uploaded PDF
    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()

    # Split the PDF into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    # Convert chunks into embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store embeddings in FAISS
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    st.success("PDF processed successfully!")

# Question box
question = st.text_input(
    "Ask a question about the PDF:"
)

# Button
if st.button("Ask"):
    if question:

        # Find the 3 most relevant chunks from the PDF
        results = vector_store.similarity_search(
            question,
            k=3
        )

        # Combine those chunks into one context
        context = "\n\n".join(
            result.page_content for result in results
        )

        # Give the retrieved information and question to Llama
        prompt = f"""
You are a helpful assistant.

Answer the question using only the context provided below.

If the answer is not available in the context, say:
"I could not find that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

        # Generate the answer
        with st.spinner("Searching the document..."):
            response = llm.invoke(prompt)

        # Show the answer
        st.subheader("Answer")
        st.write(response.content)

    else:
        st.warning("Please enter a question.")