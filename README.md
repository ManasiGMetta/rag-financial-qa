# rag-financial-qa

A Retrieval-Augmented Generation (RAG) pipeline for financial document Q&A, built with LangChain, OpenAI, and FAISS — designed around hallucination avoidance through context-grounding rather than open-ended generation.

---

## Goal

Build a RAG pipeline that answers questions about financial/banking documents (credit risk policy, loan classification, fraud detection protocols) strictly from retrieved context, and correctly declines to answer when a question falls outside what's indexed — rather than fabricating a plausible-sounding answer.

## Approach

- **Data**: Runs in **synthetic mode** by default — six pages of a fabricated internal credit-risk policy document (NPA classification, credit scoring model, fraud detection protocol, capital adequacy) so the notebook runs end-to-end without requiring a real PDF upload. Swapping in a real financial document (annual report, RBI circular, loan policy) is a one-line toggle (`SYNTHETIC_MODE = False`) plus a file path.
- **Parsing**: PyMuPDF, chosen over PyPDF2 for better handling of tables and multi-column layouts common in financial documents.
- **Chunking**: LangChain's `RecursiveCharacterTextSplitter`, which tries paragraph breaks first, then sentences, then words — preserving semantic units better than naive character-count splitting — with overlap to avoid losing context at chunk boundaries.
- **Embeddings**: OpenAI `text-embedding-3-small` (1536-dim), chosen for cost efficiency over `ada-002` at a comparable price point.
- **Vector store**: FAISS `IndexFlatL2`, in-memory, exact (brute-force) L2 search — a reasonable choice under roughly 100k vectors; noted in the code as needing a swap to Pinecone/Weaviate at larger scale. Index is persisted to disk so repeated runs don't require re-embedding.
- **Generation**: GPT-4o-mini at `temperature=0` for deterministic, cost-efficient answers, with a **context-restricted system prompt** — the model is explicitly instructed to answer only from provided context and to say "I don't know" rather than guess. Retrieved chunks are cited by page number for auditability.

## Result

- Verified the hallucination guard **qualitatively**: asked a question outside the indexed document ("What is the current interest rate on home loans?" — not present in the synthetic policy doc), and the model correctly declined rather than fabricating an answer.
- Ran three in-scope test queries (NPA classification thresholds, credit-score approval cutoffs, fraud-detection auto-block logic) and confirmed answers were grounded in the correct source pages.
- **What's not yet built, stated plainly**: there is no quantified retrieval-accuracy or faithfulness benchmark (e.g. via [RAGAS](https://github.com/explodinggradients/ragas)) — the evaluation above is qualitative, not a measured percentage. There is also no FastAPI service yet; the pipeline currently runs as a notebook. Both are listed below as concrete next steps rather than implied as already done.

## Stack

Python, LangChain, LangChain-OpenAI, LangChain-Community, OpenAI (`text-embedding-3-small` + GPT-4o-mini), FAISS, PyMuPDF, python-dotenv, Jupyter.

---

## Repository Structure

```
rag-financial-qa/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── financial_rag_pipeline.ipynb   # the full pipeline, step-by-step
```

## Quickstart

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set up your OpenAI API key**
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

**3. Run the notebook**
```bash
jupyter notebook notebooks/financial_rag_pipeline.ipynb
```
Runs end-to-end against the synthetic policy document by default — no API key needed for parsing/chunking, but embeddings and generation steps do require a valid `OPENAI_API_KEY`.

**4. Use a real document instead**
Set `SYNTHETIC_MODE = False` near the top of the notebook and point `load_pdf()` at a real financial PDF (annual report, loan policy, RBI circular, etc.).

## Honest Limitations

- **Synthetic data by default.** The six-page policy document is fabricated for demo purposes; it hasn't been validated against a real, messy financial PDF (multi-column layouts, scanned pages, embedded tables).
- **No quantified evaluation.** The hallucination guard and retrieval quality have been checked qualitatively (does it decline out-of-scope questions? does it cite the right page?) — not benchmarked with a metric like retrieval precision@k or RAGAS faithfulness score.
- **No serving layer.** This runs as a Jupyter notebook, not a deployed API. `FastAPI` is a planned next step, not a working `app/main.py` today.
- **Small-scale vector store.** FAISS `IndexFlatL2` with brute-force search is fine for the current single-document scale; it wouldn't be the right choice at production scale (see Possible Extensions).

## Possible Extensions

- Swap FAISS for Pinecone or Weaviate for cloud-scale, multi-document production use
- Add quantified evaluation using the RAGAS framework (faithfulness, answer relevancy, context precision/recall)
- Expose the pipeline as a FastAPI service for programmatic access
- Add re-ranking (e.g. Cohere Rerank) to improve retrieval precision before generation
- Add LangGraph for multi-turn conversational memory across follow-up questions
- Multi-document RAG — index multiple RBI circulars or annual reports simultaneously, rather than one document at a time

## Related Work

See [`finmulti-agent`](https://github.com/ManasiGMetta/finmulti-agent) and [`finagent`](https://github.com/ManasiGMetta/finagent), where this RAG pipeline's role is referenced as the "research" tool in a larger financial-analysis agent — currently wired as a mock-first integration point pending the FastAPI service above.
