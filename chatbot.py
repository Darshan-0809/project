import os
import re
import shutil
import streamlit as st
import pdfplumber
from docx import Document
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("Google_Api_key")

# Initialize Gemini model and embedding
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GOOGLE_API_KEY
)
model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-pro", google_api_key=GOOGLE_API_KEY
)

# Folder containing your documents
FOLDER_PATH = "Data"

# ✅ Extract text (with tables) from PDF
def get_pdf_text(file_obj):
    text = ""
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            for table in page.extract_tables():
                for row in table:
                    row_text = " | ".join(cell or "" for cell in row)
                    text += "\n" + row_text
    return text

# ✅ Extract text from PDFs, DOCX, TXT in folder
def get_all_text_from_folder(folder_path):
    combined_text = ""
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.lower().endswith(".pdf"):
            with open(filepath, "rb") as f:
                combined_text += get_pdf_text(f)
        elif filename.lower().endswith(".docx"):
            doc = Document(filepath)
            combined_text += "\n".join([para.text for para in doc.paragraphs])
        elif filename.lower().endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                combined_text += f.read()
    return combined_text

# ✅ Create and save FAISS vector index
def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")
    return db

# ✅ Load existing FAISS index
def load_vector_store():
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# ✅ Ask Gemini a question with improved strict prompt
def ask_question(db, query):
    results = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in results])

    prompt = f"""
You are a strict insurance assistant. Use ONLY the context below to answer the user's question.

🛑 Do NOT attempt to check or mention whether a specific policy name is present or not.
✅ If the context answers the question (e.g., grace period, coverage, sum insured), provide the answer directly.
🚫 Avoid saying things like:
  - "The policy is not mentioned"
  - "No such policy found"
  - Any disclaimers about policy names

🤫 If the answer is not found in the context at all, reply only:
"Insufficient information in the document."

Context:
{context}

Question: {query}

Answer:
"""

    response = model.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)

# ✅ Streamlit App UI
def main():
    st.set_page_config(page_title="Insurance Policy Q&A", layout="wide")
    st.title("📘 Chat with Your Insurance Documents")

    # 🧹 Always delete old index to avoid confusion
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")

    # 📄 Step 1: Process documents
    if not os.path.exists("faiss_index"):
        with st.spinner("📂 Processing your policy documents..."):
            text = get_all_text_from_folder(FOLDER_PATH)
            if not text.strip():
                st.error("❌ No readable content found in your documents.")
                return
            db = create_vector_store(text)
            st.success("✅ Documents processed successfully.")
    else:
        db = load_vector_store()
        st.success("✅ Loaded existing memory (faiss_index)")

    # 🧠 Step 2: User query
    query = st.text_input("💬 Ask a question about your policy:")
    if st.button("Ask") and query:
        with st.spinner("🔍 Analyzing..."):
            answer = ask_question(db, query)
            st.markdown("### ✅ Answer:")
            st.markdown(answer)

if __name__ == "__main__":
    main()
