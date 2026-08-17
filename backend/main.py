from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.rag import answer_question
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "RAG QA API is running"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    answer = answer_question(request.question)
    return {"answer": answer}