# 🧑‍⚖️ JurisProAI

An AI-powered legal assistant for querying the Indian Penal Code (IPC). Judges can manage judgments, stakeholders can upload documents, and users can ask contextual legal questions.

## 🚀 Features

- Contextual Q&A over IPC documents
- Judgment management interface
- Upload and retrieval of related legal documents
- Accessible via a clean web interface (Gradio/Streamlit)

## 🧩 Requirements

1. **Python** ≥ 3.10  
2. **Pip** for package management  
3. **Git** for cloning the repository  
4. **FAISS** integration (faiss-cpu or GPU version)  
5. **LangChain**, **Transformers**, **huggingface_hub**, **Gradio** or **Streamlit**  
6. **Pickle**, **faiss**, **numpy**, **scikit-learn**, **pandas** (depending on ingestion and IPC index)  

💡 If deploying on Hugging Face Spaces, your runtime will auto-install from `requirements.txt`, but for local runs, follow the steps below.

## 🧾 requirements.txt

```
faiss-cpu            # or faiss-gpu if using GPU acceleration
langchain
transformers
huggingface_hub
gradio               # or streamlit if you're using Streamlit
pandas
numpy
scikit-learn
```

*(Adjust this list based on actual imports in `Ingest.py` and your main app script)*

## ⚙️ Setup & Running Locally

### 1. Clone the repo

```bash
git clone https://huggingface.co/spaces/Krrish-shetty/jurisproAI
cd jurisproAI
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare or build the vector index

If you have IPC data ingestion:

```bash
python Ingest.py
```

### 5. Launch the app

For a Gradio-based app (e.g., `app.py`):

```bash
python app.py
```

If using Streamlit (e.g., `app.py`):

```bash
streamlit run app.py
```

### 6. Use the interface

- Ask questions about IPC.
- Upload judgment files.
- The app uses Indexed FAISS + embeddings to retrieve relevant content and respond.

## ⚠️ Notes

- Hugging Face Spaces: Deployment is automatic once you push to the `main` branch — make sure `requirements.txt` is present.
- Indexing: If `index.faiss` & `index.pkl` are stored in `ipc_vector_db`, ensure your code can find and load them.
- Updating dependencies: After any `pip install`, run:
  ```bash
  pip freeze > requirements.txt
  ```

## 📌 Troubleshooting

| Issue | Solution |
|-------|----------|
| FAISS import fails | Install CPU/GPU version: `pip install faiss-cpu` or `faiss-gpu` |
| Port conflicts | Set `port=XXXX` in app launch command |
| Token upload fails | Validate token via HF CLI: `huggingface-cli login` |

## ✅ To Run Locally

```bash
git clone https://huggingface.co/spaces/Krrish-shetty/jurisproAI
cd jurisproAI
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python Ingest.py       # Load or build IPC index
streamlit run app.py         # Or use streamlit run streamlit_app.py
```

Open your browser and start interacting! 👨‍⚖️