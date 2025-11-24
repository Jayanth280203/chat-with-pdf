import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# 1. Configuration
st.set_page_config(
    page_title="chat-with-pdf",
    page_icon="logo.png",  # This puts your logo in the browser tab!
    layout="wide"
)

# 2. Load API Key securely from Streamlit Secrets
# This allows the app to use YOUR key without users seeing it.
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    st.error("API Key not found. Please set it in Streamlit Secrets.")
    st.stop()

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
    # Pass the API key explicitly to the embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say "The answer is not available in the PDF context", don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    # Pass the API key explicitly to the Chat Model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    if st.session_state.vector_store is None:
        st.warning("Please upload and process a PDF first!")
        return

    docs = st.session_state.vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("🤖 **Answer:**")
    st.write(response["output_text"])

def main():
    # Display the logo. 'width' adjusts the size (try 200, 300, or 400)
    st.image("logo.png", width=300) 
    
    st.header("AI-Powered Document Conversations") # Your tagline
    st.markdown("Upload a PDF and ask questions. No API key required!")

    # Initialize session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Sidebar for Upload
    with st.sidebar:
        st.header("Upload Section")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload a file first.")
            else:
                with st.spinner("Processing PDF..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(text_chunks)
                    st.success("Done! Ask your question now.")

    # Main Chat Area
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
