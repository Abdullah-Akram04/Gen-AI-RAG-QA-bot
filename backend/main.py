import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

PDF_PATH = "document.pdf"
VECTOR_DB_PATH = "faiss_index"
DEBUG = False

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=512
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_vector_store():
    print("Loading PDF...")

    documents = PyPDFLoader(PDF_PATH).load()

    if not documents:
        raise ValueError("No pages were loaded from the PDF.")

    print(f"Loaded {len(documents)} pages.")

    total_characters = sum(
        len(doc.page_content) for doc in documents
    )

    print(f"Extracted {total_characters:,} characters.")

    if total_characters == 0:
        raise ValueError(
            "No text could be extracted from the PDF. "
            "The PDF may be scanned/image-based."
        )

    print("Splitting document into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were created from the PDF.")

    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings...")
    print("This may take some time for an 800-page PDF.")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_store.save_local(VECTOR_DB_PATH)

    print("Vector database saved successfully.")

    return vector_store


def load_vector_store():
    if Path(VECTOR_DB_PATH).exists():
        print("Loading existing vector database...")

        vector_store = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        print("Vector database loaded.")

        return vector_store

    return create_vector_store()


def answer_question(question, retriever):
    greetings = {
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there"
    }

    if question.lower().strip() in greetings:
        return "Hi! I'm ready to answer questions about your PDF."

    relevant_documents = retriever.invoke(question)

    if not relevant_documents:
        return "I could not find relevant information in the document."

    context = "\n\n".join(
        doc.page_content
        for doc in relevant_documents
    )

    if DEBUG:
        print("\n--- Retrieved Documents ---")

        for i, document in enumerate(relevant_documents, 1):
            print(f"\nChunk {i}:")
            print(document.page_content[:1000])

        print("\n--- End Retrieved Documents ---")

    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using the provided document context.

Rules:
1. Use the document context as your main source.
2. Do not invent information.
3. If the answer is not available in the context, say:
   "I could not find the answer in the document."
4. If the user asks what the document/book is about, summarize the
   relevant information available in the retrieved context.
5. Give a clear and easy-to-understand answer.

Document Context:
{context}

User Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content


def main():
    if not Path(PDF_PATH).exists():
        print(f"Error: '{PDF_PATH}' was not found.")
        print("Put your PDF in the same folder as main.py.")
        return

    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    print("\n" + "=" * 50)
    print("RAG QA BOT IS READY")
    print("=" * 50)
    print("Ask questions about your PDF.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        try:
            answer = answer_question(
                question,
                retriever
            )

            print(f"\nAssistant: {answer}\n")

        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()