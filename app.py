import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import time

st.set_page_config(page_title="chat-with-pdf", page_icon="logo.png", layout="wide")

# Securely load API Key
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    st.error("API Key not found. Please set it in Streamlit Secrets.")
    st.stop()

def get_pdf_text(pdf_docs):
    text = ""
    total_pages = 0
    progress_bar = st.progress(0, text="Reading PDF...")
    
    for file_index, pdf in enumerate(pdf_docs):
        pdf_reader = PdfReader(pdf)
        num_pages = len(pdf_reader.pages)
        total_pages += num_pages
        
        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception:
                continue
            
            # Update progress
            current_progress = (i + 1) / num_pages
            progress_bar.progress(current_progress, text=f"Reading page {i+1} of {num_pages}...")
            
    progress_bar.empty()
    return text, total_pages

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # UPDATED MODEL: "text-embedding-004" is the new standard for free tier
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    
    vector_store = None
    batch_size = 25 
    total_chunks = len(text_chunks)
    progress_bar = st.progress(0, text="Creating Knowledge Base...")
    
    for i in range(0, total_chunks, batch_size):
        batch = text_chunks[i:i + batch_size]
        try:
            if vector_store is None:
                vector_store = FAISS.from_texts(batch, embedding=embeddings)
            else:
                vector_store.add_texts(batch)
                
            progress = min((i + batch_size) / total_chunks, 1.0)
            progress_bar.progress(progress, text=f"Processing chunk {min(i + batch_size, total_chunks)} of {total_chunks}...")
            
            # Sleep to prevent hitting rate limits
            time.sleep(1) 
            
        except Exception as e:
            st.error(f"Error processing batch: {str(e)}")
            time.sleep(5) # Wait longer if error occurs
            continue
            
    progress_bar.empty()
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say "The answer is not available in the PDF context".
    
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    # FIX: Using 'gemini-1.5-flash-latest' which is safer, or 'gemini-pro' as backup
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=api_key)
    
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
    if os.path.exists("logo.png"):
        st.image("logo.png", width=300)
    else:
        st.title("📄 chat-with-pdf")
        
    st.markdown("### AI-Powered Document Conversations")
    st.markdown("Upload a PDF (even 1000+ pages!) and ask questions.")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    with st.sidebar:
        st.header("Upload Section")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload a file first.")
            else:
                with st.spinner("Processing... Do not close this tab."):
                    raw_text, page_count = get_pdf_text(pdf_docs)
                    
                    if len(raw_text) < 100:
                        st.error("Could not extract text. Is this a scanned image PDF?")
                    else:
                        st.info(f"Read {page_count} pages. Now creating AI knowledge base...")
                        text_chunks = get_text_chunks(raw_text)
                        st.session_state.vector_store = get_vector_store(text_chunks)
                        
                        if st.session_state.vector_store:
                            st.success("Done! Knowledge base created.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
