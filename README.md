#  Chat-With-PDF *using Gemini 2.0 Flash*

<img src="logo.png" alt="App Logo" width="200"/>

*A smart PDF question-answering application that allows you to upload documents and chat with them using Natural Language. Built with Python + Streamlit and powered by Google Gemini 2.0 Flash.*

🚀 Live Demo

Try it here (Free to use):
🔗 https://chat-with-pdf-jayanth.streamlit.app/

## ✨ Features

✔ Upload & analyze any PDF (notes, research papers, books, reports)

✔ Chat naturally with your document

✔ Extract summaries, tables, formulas, key points, comparisons

✔ Handles 100+ page documents efficiently

✔ Fast and optimized responses using batch embeddings

✔ Clean and responsive UI

## 🛠️ Tech Stack

✔UI Framework	-- Streamlit

✔LLM	-- Google Gemini 2.0 Flash

✔Workflow --	LangChain

✔Vector Store	-- FAISS

✔Backend Language --	Python

## 📂 Project Structure

📁 chat-with-pdf/

│── app.py                # Main Streamlit app

│── requirements.txt       # Dependencies

│── logo.png               # UI Logo

│── README.md              # Project Documentation

## 📌 Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/Jayanth280203/chat-with-pdf.git

cd chat-with-pdf

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Add Your Gemini API Key

Create a .env file and add:

GEMINI_API_KEY = your_api_key_here


(Or add it directly in Streamlit Cloud secrets if deployed.)

4️⃣ Run the App

streamlit run app.py

## 📤 Deployment

*Easily deploy on:*

   --> Streamlit Community Cloud

   --> Render

   --> Hugging Face Spaces

## 📌 Future Enhancements

🔧 Support for multiple documents

📈 Export chat history

🔍 Citation-based answers

🎙️ Add voice-based querying

📚 Support for Word, Text, and PPT files

## 🤝 Contributing

Contributions, issues, and feature requests are always welcome!
Feel free to fork this repo and submit a pull request.

## 📝 License

This project is open-source and available under the MIT License.
