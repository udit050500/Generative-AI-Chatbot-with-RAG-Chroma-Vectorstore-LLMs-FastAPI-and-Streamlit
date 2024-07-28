# Generative-AI-Chatbot-with-RAG-Chroma-Vectorstore-LLMs-FastAPI-and-Streamlit

## Overview

This project is a generative AI chatbot designed to interact with users by answering questions based on content extracted from PDF documents. It leverages Chroma Vectorstore and Azure LLM to provide accurate, concise, and contextually relevant responses. The chatbot includes features for efficient data retrieval, response generation, and document sourcing, with a robust backend powered by FastAPI and an intuitive frontend using Streamlit.

## Features

- **Interactive Q&A**: Users can ask questions about the uploaded PDF documents and receive detailed responses.
- **Efficient Data Retrieval**: Uses Chroma Vectorstore to quickly retrieve relevant information from indexed documents.
- **Accurate Response Generation**: Powered by Azure LLM, ensuring high-quality answers.
- **Conversation Memory**: Stores and retrieves chat history for a seamless user experience.
- **User-Friendly Interface**: Built with Streamlit for easy interaction.

## Tech Stack

- **Python**
- **Langchain**
- **Chroma Vectorstore**
- **Azure LLM**
- **FastAPI**
- **Streamlit**
- **PDF Processing**
- **Natural Language Processing (NLP)**
- **Conversational AI**
- **API Development**
- **Web Application Development**

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.12
- `pip` package manager

### Clone the Repository

```bash
git clone https://github.com/your-username/generative-ai-chatbot.git
cd generative-ai-chatbot

**Install Dependencies**
pip install -r requirements.txt

**### Environment Variables**

Create a .env file in the project root directory and add the following environment variables:
DEPLOYMENT_NAME=your_azure_deployment_name
AZURE_API_KEY=your_azure_api_key
AZURE_ENDPOINT=your_azure_endpoint
AZURE_API_VERSION=your_azure_api_version

**### Running the Application**

**Start the FastAPI Server**
uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8001

**Start the Streamlit Application**
Open a new terminal window and run:
streamlit run streamlit_app.py

**### Usage**

Upload PDF Files

	1.	Navigate to the Streamlit application.
	2.	Upload your PDF files using the sidebar uploader.
	3.	Click “Submit & Process” to process the PDF files.

Ask a Question

	1.	Enter your question in the text input box on the main page.
	2.	Submit your question to receive a concise answer along with the source documents.

**### Project Structure**
generative-ai-chatbot/
│
├── fastapi_app.py         # FastAPI application code
├── streamlit_app.py       # Streamlit application code
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

**### License**
This project is licensed under the MIT License. See the LICENSE file for details.

**### Acknowledgements**
	•	Langchain
	•	Chroma Vectorstore
	•	Azure Cognitive Services
	•	FastAPI
	•	Streamlit

