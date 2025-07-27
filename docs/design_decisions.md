# Design Decisions and Considerations

This document outlines the key design considerations for the semantic search microservice. It serves as a guide for making architectural and implementation choices that should be documented in the final `README.md`.

## 1. Data Handling and Preprocessing

- **Dataset Scope:**
  - Will you use the entire `meta_Amazon_Fashion.jsonl` dataset (1.3GB) or a smaller subset for development and demonstration? The assignment mentions a cloud credit, implying that using the full dataset is possible but not required.
  - **Decision:** Start with a subset (e.g., the first 100,000 products) for faster iteration and switch to the full dataset for final evaluation.

- **Data Storage:**
  - How will the product metadata and their embeddings be stored?
  - **Options:**
    - **In-Memory:** Load everything into RAM (e.g., using Pandas DataFrames and NumPy arrays). Simple and fast for smaller datasets but not scalable.
    - **File-Based:** Store processed data and embeddings in files (e.g., Parquet for metadata, FAISS index for vectors).
    - **Vector Database:** Use a dedicated vector database (e.g., FAISS, ChromaDB, Pinecone) for efficient storage and retrieval of embeddings. This is a more scalable, production-oriented approach.

- **Text Preprocessing:**
  - Which text fields will be used to generate embeddings (`title`, `description`, `features`, `categories`)?
  - How will these fields be combined into a single text document for embedding? Simple concatenation or a more structured format?
  - Will there be any text cleaning (e.g., removing HTML tags, lowercasing)?

## 2. Semantic Search and AI/ML Model Selection

- **Embedding Model:**
  - Which model will be used to convert product text into vector embeddings?
  - **Options:**
    - **Sentence-Transformers (Hugging Face):** Open-source models like `all-MiniLM-L6-v2` or `all-mpnet-base-v2`. They are fast, run locally, and offer a good performance baseline.
    - **OpenAI Embeddings:** Use a service like `text-embedding-ada-002`. This requires an API key and incurs costs but often provides high-quality embeddings.
    - **Multimodal Models:** For future consideration, models like CLIP could be used to embed both text and images for a richer search experience.

- **Vector Search (Nearest Neighbor) Algorithm:**
  - How will you find the most similar products in the vector space?
  - **Options:**
    - **Brute-Force Search:** Calculate the cosine similarity between the query vector and all product vectors. It's exact but slow and not feasible for large datasets.
    - **Approximate Nearest Neighbor (ANN):** Use libraries like FAISS or Annoy for a much faster, approximate search. This is the standard for large-scale vector search.

- **LLM Integration:**
  - How will a Large Language Model (LLM) be used in the service?
  - **Options:**
    - **Query Rewriting:** Use an LLM to transform the user's conversational query (e.g., "outfit for the beach") into a more descriptive text prompt that can be embedded for search.
    - **Re-ranking & Summarization:** Use the LLM to re-rank the top K results from the vector search or to generate a natural language explanation for why a product is a good match.
    - **Model Choice:** Will you use the OpenAI API (e.g., GPT-4o, GPT-3.5-turbo) or a smaller, open-source model?

## 3. System Architecture and API Design

- **Service Interface:**
  - How will the service's functionality be exposed? The requirements list three options.
  - **Options:**
    - **Function Call:** The simplest approach, packaging the logic into a callable Python function.
    - **Command-Line Interface (CLI):** A good way to demonstrate functionality without the overhead of a web server. Libraries like `Typer` or `argparse` can be used.
    - **API Endpoint:** The most realistic approach for a microservice. A lightweight web framework like **FastAPI** or **Flask** is ideal.

- **API Specification (if chosen):**
  - What will be the request and response format?
  - **Example Request:** `POST /recommend` with a JSON body: `{"query": "a dress for a summer wedding", "top_k": 5}`
  - **Example Response:** JSON array of recommended products: `[{"parent_asin": "B01CUPMQZE", "title": "..."}]`

- **Code Structure:**
  - How will the codebase be organized?
  - **Suggestion:** A modular structure is recommended.
    - `data/`: For data files.
    - `notebooks/`: For exploration (e.g., `pilot.ipynb`).
    - `src/` or `search_engine/`: For the main application logic.
      - `data_processing.py`: To load and clean data.
      - `embedding.py`: To handle text embedding logic.
      - `search.py`: For the vector search implementation.
      - `main.py`: To expose the CLI or API.

## 4. Future Enhancements (Design Considerations)

- **Using User Reviews:** How could the user review data be incorporated in the future? (e.g., for sentiment analysis, fine-tuning an embedding model, or as a re-ranking signal).
- **Multimodality:** How could you extend the system to handle image-based search?
- **Scalability:** How would you scale the service to handle millions of products and high traffic? (e.g., managed vector database, caching layers, horizontal scaling of the microservice).
- **UI:** Adding a minimal front-end (e.g., with Streamlit or Gradio) to create a more interactive demo. 