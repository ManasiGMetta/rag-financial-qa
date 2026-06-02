# 📊 Financial Document Q&A — RAG Pipeline

A production-style **Retrieval-Augmented Generation (RAG)** system for querying financial and banking documents using natural language.

Built to demonstrate end-to-end GenAI engineering skills: document ingestion, vector search, LLM integration, and hallucination reduction.

---

## 🏗️ Architecture

```
PDF Document
     │
     ▼
Text Extraction (PyMuPDF)
     │
     ▼
Chunking (LangChain RecursiveCharacterTextSplitter)
     │
     ▼
Embeddings (OpenAI text-embedding-3-small · 1536 dims)
     │
     ▼
Vector Store (FAISS IndexFlatL2)
     │
     ▼
Retriever (Top-K Semantic Similarity Search)
     │
     ▼
LLM (GPT-4o-mini · temperature=0) + Prompt Engineering
     │
     ▼
Grounded Answer with Source Page Citations
```

---

## ✨ Features

- **PDF ingestion** — handles multi-page financial documents (annual reports, RBI circulars, loan policy docs, credit risk frameworks)
- **Semantic chunking** — recursive splitting that preserves paragraph and sentence boundaries
- **OpenAI embeddings** — `text-embedding-3-small` for cost-efficient, high-quality vectors
- **FAISS vector index** — in-memory similarity search with save/load to avoid re-embedding
- **Hallucination guard** — system prompt restricts LLM to context only; explicitly declines out-of-scope questions
- **Source citations** — every answer cites the source page number(s)
- **Interactive Q&A loop** — ask questions in a terminal session
- **Runs without a PDF** — synthetic financial document included for immediate demonstration

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector DB | FAISS (Facebook AI Similarity Search) |
| Orchestration | LangChain |
| PDF Parsing | PyMuPDF (fitz) |
| API Framework | FastAPI *(extension — see roadmap)* |

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/rag-financial-qa.git
cd rag-financial-qa
pip install -r requirements.txt
```

### 2. Set your OpenAI API key
```bash
# Create a .env file in the project root
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### 3. (Optional) Add your own PDF
```bash
# Place any financial PDF here:
cp your_document.pdf data/financial_document.pdf
```

### 4. Run the notebook
```bash
jupyter notebook notebooks/financial_rag_pipeline.ipynb
```

Run all cells top to bottom. The notebook includes a **synthetic financial document** so it works out-of-the-box without a PDF.

---

## 📁 Project Structure

```
rag-financial-qa/
│
├── notebooks/
│   └── financial_rag_pipeline.ipynb   ← Main RAG pipeline (start here)
│
├── data/
│   └── financial_document.pdf         ← Place your PDF here (gitignored)
│
├── outputs/
│   └── faiss_financial_index/         ← Saved FAISS index (gitignored)
│
├── .env.example                        ← API key template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 💬 Example Queries & Answers

**Q: What is the classification for a loan overdue between 31 to 90 days?**
> A loan overdue between 31 to 90 days is classified as a **Sub-Standard Asset**. *(Source: Page 2)*

**Q: What credit score is required for automatic loan approval?**
> Applicants with a model score **above 700** are auto-approved. Scores between 600–700 go to manual review, and below 600 are automatically declined. *(Source: Page 3)*

**Q: What is the current home loan interest rate?**
> This information is not available in the provided document. *(Hallucination guard working correctly)*

---

## 🧩 Key Design Decisions

**Why FAISS over Pinecone/Weaviate?**
FAISS runs entirely in-memory with no external service or API key. Ideal for prototyping and single-node deployments. For production scale (>100k documents), Pinecone or Weaviate would be preferred.

**Why temperature=0 for the LLM?**
Financial Q&A requires deterministic, factual responses. Temperature=0 eliminates creative variation and ensures consistent answers for the same query.

**Why chunk_overlap=100?**
Prevents loss of context at chunk boundaries — particularly important for financial documents where a sentence spanning two chunks (e.g. a loan condition) could be split and lose meaning.

---

## 🗺️ Roadmap / Possible Extensions

- [ ] Expose as **FastAPI REST endpoint** (`POST /query`)
- [ ] Add **LangGraph** for multi-turn conversational memory
- [ ] Add **RAGAS evaluation** (faithfulness, answer relevancy, context recall)
- [ ] Support **multi-document indexing** (multiple RBI circulars, multiple annual reports)
- [ ] Add **Cohere Rerank** for improved retrieval precision
- [ ] Deploy to **Azure Container Apps** or AWS Lambda

---

## 👤 Author

**[Your Name]**  
MSc Advanced Computer Science — University of Manchester  
10+ years in AI/ML | Data Science | FinTech domain  

---

## 📄 License

MIT License
