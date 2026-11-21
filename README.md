# Academic Research Assistant

A small Streamlit-based retrieval-augmented generation (RAG) application for asking questions about the academic papers bundled in this repository. It retrieves relevant papers from a local FAISS index, optionally filters them by publication year, and asks Gemini to answer using only the retrieved context.

## What it does

1. Loads paper records from `data/academic_papers.csv`.
2. Searches the pre-built FAISS index in `data/faiss_index/` using configurable embeddings.
3. Re-ranks matches using a weighted combination of semantic similarity (80%) and publication recency (20%).
4. Sends the top five papers to Gemini with instructions to avoid unsupported claims and cite paper titles and years.
5. Presents the answer through a Streamlit interface with a minimum publication-year filter.

The included dataset contains 57 synthetic academic papers spanning 2017–2026 across areas including cybersecurity, machine learning, natural language processing, and quantum computing.

## Requirements

- Python 3.11 or 3.12
- A Google API key for answer generation (`GOOGLE_API_KEY`)
- An embedding API key:
  - `GOOGLE_API_KEY` when using Gemini embeddings
  - `OPENAI_API_KEY` when using OpenAI embeddings

The project declares its dependencies in `pyproject.toml` and also includes a `requirements.txt` compatibility list.

## Setup

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or using a standard virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Create a `.env` file in the repository root:

```dotenv
# auto selects OpenAI if OPENAI_API_KEY exists; otherwise Gemini
AI_PROVIDER=gemini
GOOGLE_API_KEY=your-google-api-key
```

To use OpenAI embeddings instead, provide both keys because answer generation currently uses Gemini:

```dotenv
AI_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-api-key
```

`AI_PROVIDER` accepts `auto`, `gemini`, or `openai`. In `auto` mode, OpenAI is selected when `OPENAI_API_KEY` is present; otherwise Gemini is selected. Missing or invalid configuration raises an explicit error.

## Run the application

From the repository root:

```bash
uv run streamlit run src/project/app.py
```

With an activated virtual environment, use:

```bash
streamlit run src/project/app.py
```

Open the local URL shown by Streamlit, enter a question, choose the minimum publication year, and select **Search**.

## Rebuild the FAISS index

A pre-built index is checked in under `data/faiss_index/`. Rebuild it after changing `data/academic_papers.csv` or changing the embedding provider:

```bash
uv run python -c "from project.vectorstore import build_vectorstore; build_vectorstore()"
```

The command reads the CSV, creates embeddings, and overwrites `data/faiss_index/index.faiss` and `data/faiss_index/index.pkl`. The index must be rebuilt with the same embedding provider that will be used for searches.

## Project layout

```text
data/
├── academic_papers.csv       # Source paper dataset
├── documents.json             # Additional sample documents
└── faiss_index/               # Persisted FAISS index and metadata

src/project/
├── app.py                     # Streamlit UI
├── config.py                  # Environment and provider selection
├── embeddings.py              # Gemini/OpenAI embedding factory
├── loader.py                  # CSV-to-LangChain document loader
├── rag.py                     # Retrieval and Gemini answer generation
├── retriever.py               # FAISS search and re-ranking
└── vectorstore.py             # Index construction and persistence

main.py                         # Simple data/provider smoke check
test_search.py                  # Manual retrieval smoke script
test_rag.py                     # Manual end-to-end RAG smoke script
test_gemini.py                  # Gemini embedding smoke script
```

## Manual checks

These files are executable smoke scripts rather than a configured pytest suite:

```bash
uv run python test_search.py
uv run python test_rag.py
uv run python test_gemini.py
```

They require the relevant API keys and a working embedding index. `test_rag.py` invokes the live Gemini model and may incur API usage.

## Notes and limitations

- The answer-generation model is configured directly in `src/project/rag.py` as `gemini-3.5-flash-lite`; `AI_PROVIDER` controls embeddings, not the chat model.
- Retrieval loads the local FAISS index with `allow_dangerous_deserialization=True`; only use an index generated from a trusted source.
- The application answers from the bundled dataset only. It does not fetch current papers or provide a general web search.
- The repository contains synthetic paper content intended for experimentation, not authoritative academic research.

## License

No license file is currently included in the repository.
