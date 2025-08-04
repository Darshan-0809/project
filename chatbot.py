import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import extract_msg
from newspaper import Article
import os
from dotenv import load_dotenv
import json
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -------- File Extraction Functions --------
def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_msg(file):
    msg = extract_msg.Message(file)
    return msg.body

def extract_text_from_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def get_combined_text(files, url_text):
    full_text = ""
    for file in files:
        if file.name.endswith(".pdf"):
            full_text += extract_text_from_pdf(file)
        elif file.name.endswith(".docx"):
            full_text += extract_text_from_docx(file)
        elif file.name.endswith(".msg"):
            full_text += extract_text_from_msg(file)
    full_text += url_text
    return full_text

# -------- Chunking & Embedding --------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store = FAISS.from_texts(text_chunks, embedding=embeddings)
    store.save_local("faiss_index")

# -------- Query Parsing --------
def parse_query(user_query):
    prompt_template = PromptTemplate(
        input_variables=["query"],
        template="""
Extract the following fields from the query: age, gender, procedure, location, and policy duration.
If not available, leave them as null.

Query:
{query}

Respond in JSON:
{{
  "age": ...,
  "gender": "...",
  "procedure": "...",
  "location": "...",
  "policy_duration": "..."
}}
""")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    chain = LLMChain(llm=model, prompt=prompt_template)
    result = chain.run(query=user_query)
    return result

# -------- Decision Making Chain --------
def get_decision_chain():
    prompt_template = PromptTemplate(
        input_variables=["structured_query", "context"],
        template="""
You are an expert insurance policy analyzer. Given the user's structured query and the context (retrieved clauses), determine whether the procedure is covered.

Structured Query:
{structured_query}

Context:
{context}

Provide response in this exact JSON format:
{{
  "decision": "...",       // approved / rejected / unclear
  "amount": "...",         // in ₹ or USD or "N/A"
  "justification": "...",  // why the decision was made
  "clauses_used": ["..."]  // list of clauses or section titles used
}}
""")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    return LLMChain(llm=model, prompt=prompt_template)

# -------- Clean & Display Result --------
def clean_json_output(raw_output):
    raw_output = re.sub(r"```json|```", "", raw_output).strip()
    json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return raw_output

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = db.similarity_search(user_question, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    structured_query = parse_query(user_question)
    
    decision_chain = get_decision_chain()
    result = decision_chain.run({
        "structured_query": structured_query,
        "context": context
    })

    st.subheader("📋 Structured JSON Output")
    st.code(result, language='json')

    try:
        cleaned = clean_json_output(result)
        result_dict = json.loads(cleaned)

        readable_response = (
            f"The procedure has been **{result_dict['decision']}**"
            f"{' with a payout of ' + result_dict['amount'] if result_dict['amount'] != 'N/A' else ''}. "
            f"This decision is based on {', '.join(result_dict['clauses_used'])}, "
            f"which state: *{result_dict['justification']}*"
        )

        st.markdown("### 🧾 Natural Language Explanation")
        st.markdown(readable_response)

    except Exception as e:
        st.error("Error parsing JSON: " + str(e))
        st.text("Model output was:")
        st.write(result)

# -------- Streamlit App --------
def main():
    st.set_page_config(page_title="Policy Decision AI")
    st.header("🤖 Chat with Insurance Policy Docs")

    user_query = st.text_input("Enter your case description (e.g., '46-year-old male, knee surgery in Pune, 3-month-old policy')")
    if user_query:
        user_input(user_query)

    with st.sidebar:
        st.title("📂 Upload Files & URLs")
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or Email files",
            accept_multiple_files=True,
            type=["pdf", "docx", "msg"]
        )

        url_text = ""
        url = st.text_input("🔗 Website URL")
        if st.button("Fetch Website Content"):
            with st.spinner("Fetching content..."):
                try:
                    url_text = extract_text_from_url(url)
                    st.success("✅ Website content fetched.")
                except Exception as e:
                    st.error(f"Failed: {e}")

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                combined_text = get_combined_text(uploaded_files, url_text)
                chunks = get_text_chunks(combined_text)
                get_vector_store(chunks)
                st.success("✅ Documents processed. Ask your question now.")

if __name__ == "__main__":
    main()