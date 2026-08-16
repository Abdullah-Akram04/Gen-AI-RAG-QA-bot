import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def llm_model(model_id):

    llm = ChatGroq(
        model=model_id,
        temperature=0.5,
        max_tokens=256
    )

    return llm


llm = llm_model("llama-3.1-8b-instant")

response = llm.invoke(
    "Explain Retrieval Augmented Generation in simple words."
)

print(response.content)