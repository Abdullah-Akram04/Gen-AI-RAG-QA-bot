from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "document.pdf"
VECTOR_DB_PATH = BASE_DIR / "faiss_index"
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

    documents = PyPDFLoader(str(PDF_PATH)).load()

    if not documents:
        raise ValueError("No pages were loaded from the PDF.")

    print(f"Loaded {len(documents)} pages.")

    total_characters = sum(
        len(doc.page_content) for doc in documents
    )

    if total_characters == 0:
        raise ValueError(
            "No text could be extracted from the PDF."
        )

    print(f"Extracted {total_characters:,} characters.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were created.")

    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings...")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_store.save_local(str(VECTOR_DB_PATH))

    print("Vector database saved.")

    return vector_store


def load_vector_store():
    if VECTOR_DB_PATH.exists():
        print("Loading existing vector database...")

        return FAISS.load_local(
            str(VECTOR_DB_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )

    return create_vector_store()


vector_store = load_vector_store()

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)


def answer_question(question: str) -> str:
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
        document.page_content
        for document in relevant_documents
    )

    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using the provided document context.

Rules:
1. Use the document context as your main source.
2. Do not invent information.
3. If the answer is not available in the context, say:
   "I could not find the answer in the document."
4. If the user asks what the document is about,
   summarize the relevant information from the context.
5. Give a clear and easy-to-understand answer.

Document Context:
{context}

User Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content