import os
import json
import streamlit as st
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

# LangChain Imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings

# Google GenAI SDK Resmi
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Custom Embedding Wrapper Menggunakan SDK `google-genai` (v1 Stable)
# ---------------------------------------------------------
class SafeGoogleEmbeddings(Embeddings):
    """Wrapper kustom berbasis SDK google-genai dengan paksaan API version 'v1'."""
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY tidak ditemukan di file .env")
        
        # Paksa penggunaan API version v1 (stable)
        self.client = genai.Client(
            api_key=api_key,
            http_options={'api_version': 'v1'}
        )

    def _embed_single_text(self, text: str, task_type: str) -> list[float]:
        """Memanggil API embedding dengan fallback kandidat model resmi."""
        # Daftar model embedding yang valid pada API v1
        models_to_try = ["gemini-embedding-001", "models/text-embedding-004", "embedding-001"]
        last_exception = None

        for model_name in models_to_try:
            try:
                response = self.client.models.embed_content(
                    model=model_name,
                    contents=text,
                    config=types.EmbedContentConfig(task_type=task_type)
                )
                if response.embeddings and len(response.embeddings) > 0:
                    return response.embeddings[0].values
            except Exception as e:
                last_exception = e
                continue

        raise Exception(f"Gagal memanggil model embedding: {str(last_exception)}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            if not text.strip():
                continue
            emb = self._embed_single_text(text, task_type="RETRIEVAL_DOCUMENT")
            embeddings.append(emb)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embed_single_text(text, task_type="RETRIEVAL_QUERY")

# ---------------------------------------------------------
# Konfigurasi Halaman Streamlit
# ---------------------------------------------------------
st.set_page_config(page_title="AI Chatbot PDF (Gemini API)", page_icon="📚", layout="wide")
st.title("📚 Chatbot AI Berbasis PDF (Google Gemini)")

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# ---------------------------------------------------------
# Parsing List Model dari .env
# ---------------------------------------------------------
raw_models = os.getenv("GOOGLE_MODEL", '["gemini-1.5-flash", "gemini-1.5-pro"]')
try:
    model_list = json.loads(raw_models)
    if not isinstance(model_list, list):
        model_list = [str(model_list)]
except Exception:
    model_list = [m.strip(" []\"'") for m in raw_models.split(",") if m.strip()]

# ---------------------------------------------------------
# Sidebar UI: Konfigurasi & Upload PDF
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Konfigurasi System")
    selected_model = st.selectbox("Pilih Model AI Gemini:", options=model_list)
    
    st.divider()
    st.header("📁 Kelola Dokumen PDF")
    
    # Form upload file PDF
    with st.form("upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Upload file PDF baru:", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        submit_button = st.form_submit_button("💾 Simpan File PDF")

    if submit_button:
        if uploaded_files:
            saved_count = 0
            for uploaded_file in uploaded_files:
                file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_count += 1
            st.success(f"Berhasil menyimpan {saved_count} file ke folder `{DATA_FOLDER}/`!")
            st.cache_resource.clear()
            st.rerun()
        else:
            st.warning("Pilih minimal satu file PDF terlebih dahulu.")

    st.divider()
    if st.button("🔄 Reload & Re-index Data"):
        st.cache_resource.clear()
        st.success("Indeks berhasil diperbarui!")
        st.rerun()

    existing_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".pdf")]
    st.subheader(f"Daftar File PDF ({len(existing_files)}):")
    if existing_files:
        for f in existing_files:
            st.caption(f"• {f}")
    else:
        st.caption("Belum ada file PDF di folder `data/`.")

# ---------------------------------------------------------
# Fungsi Inisialisasi Vector Store (FAISS)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Memproses & mengindeks seluruh file PDF...")
def initialize_vector_store():
    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        return None

    loader = PyPDFDirectoryLoader(DATA_FOLDER)
    raw_documents = loader.load()

    if not raw_documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(raw_documents)

    try:
        # Panggil kustom embedding yang terhubung langsung ke SDK google-genai
        embeddings = SafeGoogleEmbeddings()
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Gagal menginisialisasi model Embedding Gemini: {str(e)}")
        return None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

vector_store = initialize_vector_store()

# ---------------------------------------------------------
# Tampilan Utama & Chat Logic (LCEL Architecture)
# ---------------------------------------------------------
if vector_store is None:
    st.info("💡 **Belum ada file PDF di dalam sistem.**")
    st.write("Silakan upload file PDF melalui **Sidebar di sebelah kiri** untuk mulai menggunakannya sebagai basis pengetahuan chatbot.")
else:
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model=selected_model, temperature=0.3)

    template = """Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan dokumen referensi yang diberikan.
Gunakan potongan konteks berikut untuk menjawab pertanyaan. Jika Anda tidak mengetahui jawabannya, katakan secara jujur bahwa informasi tidak ditemukan dalam dokumen.
Jawablah dengan ringkas, jelas, dan menggunakan Bahasa Indonesia yang santun.

Konteks Dokumen:
{context}

Pertanyaan: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Tanyakan sesuatu tentang dokumen PDF Anda..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner(f"Memproses jawaban menggunakan model `{selected_model}`..."):
                try:
                    answer = rag_chain.invoke(user_input)
                except Exception as e:
                    answer = f"⚠️ Terjadi kesalahan saat memproses permintaan: {str(e)}"
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})