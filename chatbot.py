import os
import requests
import tempfile
import pdfplumber
import shutil
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

# Load env variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HACKRX_API_KEY = os.getenv("HACKRX_API_KEY")

# Auth setup
auth_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.credentials != HACKRX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return credentials.credentials

# Gemini setup
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
model = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", google_api_key=GOOGLE_API_KEY)

# FastAPI setup
app = FastAPI(
    title="HackRx Insurance Policy QA API",
    description="Submit a policy document URL and questions. Returns structured answers from Gemini 2.5 Pro.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response schema
class QARequest(BaseModel):
    documents: str
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

# Extract text and tables
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            for table in page.extract_tables():
                for row in table:
                    row_text = " | ".join(cell or "" for cell in row)
                    text += "\n" + row_text
    return text

# Run /hackrx/run
@app.post("/hackrx/run", response_model=QAResponse)
async def run_qa(request: QARequest, token: str = Depends(verify_token)):
    # Download file
    try:
        response = requests.get(request.documents, stream=True)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download document.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        # Extract text
        raw_text = extract_text_from_pdf(tmp_path)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in PDF.")

        # Create vector DB
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = splitter.create_documents([raw_text])
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")
        db = FAISS.from_documents(documents, embeddings)

        # Process questions
        answers = []
        for question in request.questions:
            results = db.similarity_search(question, k=3)
            context = "\n".join([doc.page_content for doc in results])

            prompt = f"""
You are a strict insurance assistant. Use ONLY the context below to answer the user's question.

🛑 Do NOT mention missing policy names or give disclaimers.
✅ If context contains the answer, give it directly.
🚫 Avoid saying things like:
  - "Policy not mentioned"
  - "No such policy found"

🤫 If no info found, say:
"Insufficient information in the document."

Context:
{context}

Question: {question}

Answer:
"""
            result = model.invoke(prompt)
            final_answer = result.content.strip() if hasattr(result, "content") else str(result).strip()
            if not final_answer:
                final_answer = "Insufficient information in the document."
            answers.append(final_answer)

        return {"answers": answers}

    finally:
        os.remove(tmp_path)

