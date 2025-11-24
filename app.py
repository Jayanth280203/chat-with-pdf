import streamlit as st
from pypdf import PdfReader  # UPDATED: Using the modern library
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

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
    # Create a progress bar
    progress_bar = st.progress(0)
    
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        num_pages = len(pdf_reader.pages)
        total_pages += num_pages
        
        # Iterate through pages with a progress update
        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text: # Only add if text was found
                    text += page_text + "\n"
            except Exception as e:
                # If a page fails, skip it but keep going
                continue
                
            # Update progress bar (visual feedback for large files)
            progress = (i + 1) / num_pages
            progress_bar.progress(progress)
            
    progress_bar.empty() # Remove bar when done
    return text, total_pages

def get_text_chunks(text):
    # Chunk size 1000 is safe for Free Tier limits
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        # Create vector store
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating knowledge base: {e}")
        return None

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say "The answer is not available in the PDF context".
    
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
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
    # Logo and Header
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
                with st.spinner("Reading & Processing... This may take time for large files."):
                    # Get text and page count
                    raw_text, page_count = get_pdf_text(pdf_docs)
                    
                    if len(raw_text) < 100:
                        st.error("Could not extract text. Is this a scanned image PDF? (OCR not supported in free version)")
                    else:
                        st.info(f"Successfully extracted text from {page_count} pages.")
                        text_chunks = get_text_chunks(raw_text)
                        
                        # Process chunks
                        st.session_state.vector_store = get_vector_store(text_chunks)
                        st.success("Done! You can now chat.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
