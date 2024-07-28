# fastapi_app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

app = FastAPI()

# Initialize LLM
llm = AzureChatOpenAI(
    model=os.getenv("DEPLOYMENT_NAME"),
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENPOINT"),
    api_version=os.getenv("AZURE_API_VERSION")
)

# Initialize Chat history memory once and persist state
memory = ConversationBufferMemory(memory_key="chat_history", input_key="input", output_key="output", return_messages=True)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory="chroma_local_index")

@app.post("/process_pdfs")
async def process_pdfs(files: list[UploadFile] = File(...)):
    pdf_docs = [file.file for file in files]
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    return {"status": "success"}

@app.post("/rag_q&a")
async def user_input(user_question: str = Form(...)):
    embeddings = HuggingFaceEmbeddings()
    new_db_1 = Chroma(persist_directory="chroma_local_index", embedding_function=embeddings).as_retriever()
    history_aware_retriever = create_history_aware_retriever(llm, new_db_1, contextualize_q_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Load chat history from memory
    chat_history = memory.load_memory_variables(inputs={})["chat_history"]
    response_1 = rag_chain.invoke({"input": user_question, "chat_history": chat_history})
    
    # Save user question and AI response to chat history
    memory.save_context(inputs={"input": user_question}, outputs={"output": response_1["answer"]})
    
    # Extract source document content
    source_documents = [{"page_content": doc.page_content} for doc in response_1["context"]]

    return JSONResponse(content={"reply": response_1["answer"], "source_documents": source_documents})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)