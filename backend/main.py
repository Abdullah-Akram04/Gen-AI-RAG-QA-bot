from fastapi import FastAPI
from pydantic import BaseModel
from backend.rag import answer_question
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "RAG QA API is running"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    answer = answer_question(request.question)
    return {"answer": answer}