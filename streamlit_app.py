# streamlit_app.py
import streamlit as st
import requests

st.set_page_config("Chat PDF")
st.header("Chat with PDFs (GenerativeAI Chatbot with RAG, Chroma Vectorstore, and LLMs)")

# FastAPI endpoints
FASTAPI_URL = "http://localhost:8001"

# Upload PDF files
with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    if st.button("Submit & Process"):
        if pdf_docs:
            with st.spinner("Processing..."):
                files = [("files", pdf) for pdf in pdf_docs]
                response = requests.post(f"{FASTAPI_URL}/process_pdfs", files=files)
                if response.status_code == 200:
                    st.success("PDF files processed successfully.")
                else:
                    st.error("Failed to process PDF files.")

# Ask a question
user_question = st.text_input("Ask a Question from the PDF Files")

if user_question:
    with st.spinner("Fetching answer..."):
        response = requests.post(f"{FASTAPI_URL}/rag_q&a", data={"user_question": user_question})
        if response.status_code == 200:
            answer = response.json().get("reply")
            source = response.json().get("source_documents")
            st.write("Reply: ", answer)
            st.write("Source Documents: ", source)
        else:
            st.error("Failed to fetch the answer.")
