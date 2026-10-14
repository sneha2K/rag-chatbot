import os
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

pdf_path = "documents/attention_is_all_you_need.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

# print("Number of pages:", len(documents))

# print("\nFirst page content:")
# print(documents[0].page_content[:500])
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0].page_content)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector database created successfully!")
question = input("\nAsk a question about the PDF: ")

results = vector_store.similarity_search(
    question,
    k=3
)

print("\nQuestion:", question)
print("\nTop relevant chunks:")

for i, result in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(result.page_content)

# Create the LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# Combine the retrieved chunks
context = "\n\n".join(
    result.page_content for result in results
)

# Create the prompt
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

# Send the prompt to the LLM
response = llm.invoke(prompt)

print("\n===== FINAL ANSWER =====")
print(response.content)