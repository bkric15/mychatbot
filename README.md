# 📚 AI PDF Chatbot with Streamlit & Google Gemini

A modern RAG (Retrieval-Augmented Generation) Chatbot application built using **Streamlit**, **LangChain**, and **Google Gemini AI**. This application allows users to interact with multiple PDF documents placed in a designated data directory or uploaded through a user-friendly UI.

---

## ✨ Features

- 📄 **Multi-PDF Document Processing**: Load and index multiple PDF files seamlessly.
- 📤 **UI-based File Upload**: Upload PDF files directly from the Streamlit sidebar without manually copying files.
- 🔄 **Real-Time Data Reload**: Dynamic re-indexing of documents with a click of a button.
- 🤖 **Dynamic Model Selection**: Switch between different Google Gemini models (e.g., `gemini-1.5-flash`, `gemini-1.5-pro`) on the fly via `.env` configuration.
- ⚡ **High-Performance Embedding**: Built using Google's modern `google-genai` SDK with custom fallback handling and FAISS vector database.
- 💬 **Interactive Chat Interface**: Conversational experience with memory within the active session.

---

## 🏗️ System Architecture

```text
  [ User Interface (Streamlit) ]
                 │
  ┌──────────────┴──────────────┐
  │  Sidebar Controls           │
  │  - Model Selector           │
  │  - PDF File Uploader        │
  │  - Reload / Re-index Button │
  └──────────────┬──────────────┘
                 │
                 ▼
     [ PDF Loader & Splitter ]  ──────► Save files to 'data/'
                 │
                 ▼
     [ Google GenAI Embeddings ]
                 │
                 ▼
       [ FAISS Vector Store ]
                 │
                 ▼
    [ LangChain LCEL RAG Pipeline ] ◄──► [ Google Gemini LLM ]
```

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### 📋 Prerequisites

- **Python**: Version `3.9` or higher.
- **Google Gemini API Key**: Obtain a key from [Google AI Studio](https://aistudio.google.com/).

---

### 📥 Installation & Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/bkric15/mychatbot.git
cd your-repo-name
```

#### 2. Create and Activate a Virtual Environment

* **Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

### ⚙️ Environment Configuration

Create a `.env` file in the root directory of the project:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
PORT=8501
GOOGLE_MODEL=["gemini-1.5-flash", "gemini-1.5-pro"]
```

> **Note:** Ensure `GOOGLE_MODEL` is written as a valid JSON string array or comma-separated list.

---

## 🏃 Running the Application

Execute the following command to launch the Streamlit application:

```bash
streamlit run app.py
```

Or run using the specific port defined in your `.env`:

```bash
streamlit run app.py --server.port 8501
```

Once running, access the web UI at `http://localhost:8501`.

---

## 📖 Usage Guide

1. **Upload Documents**: Open the sidebar on the left and upload one or multiple `.pdf` files. Click **"💾 Simpan File PDF"**.
2. **Select AI Model**: Choose your preferred Gemini LLM model from the dropdown list.
3. **Ask Questions**: Type your questions into the chat input at the bottom. The AI will retrieve relevant contexts from your PDF files to generate answers.
4. **Reload Index**: If you manually add or remove files in the `data/` directory, click **"🔄 Reload & Re-index Data"** in the sidebar.

---

## 📁 Project Structure

```text
.
├── data/               # Auto-created folder storing all uploaded PDF files
├── .env                # Environment variables configuration
├── app.py              # Main application entry point
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## 🛠️ Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `GOOGLE_API_KEY Error` | Missing or invalid API key in `.env`. | Verify your key in [Google AI Studio](https://aistudio.google.com/) and paste it in `.env`. |
| `Embedding Model 404` | Outdated SDK or endpoint routing issue. | Ensure you are using the `google-genai` package and latest `app.py` wrapper. |
| `No Documents Loaded` | The `data/` folder is empty. | Upload at least one PDF file using the UI sidebar or manually place PDFs in `data/`. |

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
