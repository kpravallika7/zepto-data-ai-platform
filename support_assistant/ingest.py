from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ================================================================
# MODULE 3 - DOCUMENT INGESTION
# ================================================================

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_support"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_documents():
    """Load all text documents from the docs directory."""

    documents = []

    for file_path in sorted(DOCS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        documents.append(
            {
                "id": file_path.stem,
                "text": text,
                "source": file_path.name,
            }
        )

    return documents


def chunk_documents(documents):
    """
    Create chunks from documents.

    The Zepto documents are short, so one chunk per document
    is sufficient for this task.
    """

    chunks = []

    for document in documents:
        chunks.append(
            {
                "id": document["id"],
                "text": document["text"],
                "source": document["source"],
            }
        )

    return chunks


def create_embeddings(chunks):
    """Generate local embeddings using all-MiniLM-L6-v2."""

    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    return embeddings


def store_in_chromadb(chunks, embeddings):
    """Store documents and embeddings in a ChromaDB collection."""

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Recreate the collection so repeated ingestion remains deterministic.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Zepto customer support documents",
            "hnsw:space": "cosine",
        },
    )

    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": chunk["source"]}
            for chunk in chunks
        ],
    )

    return collection


def main():
    print("=" * 60)
    print("MODULE 3 - DOCUMENT INGESTION")
    print("=" * 60)

    # 1. Load documents
    documents = load_documents()

    print(f"\nDocuments loaded: {len(documents)}")

    for document in documents:
        print(f"  - {document['source']}")

    if len(documents) != 8:
        raise ValueError(
            f"Expected 8 documents, found {len(documents)}"
        )

    # 2. Chunk documents
    chunks = chunk_documents(documents)

    print(f"\nChunks created: {len(chunks)}")

    # 3. Generate embeddings
    embeddings = create_embeddings(chunks)

    print(f"\nEmbedding shape: {embeddings.shape}")

    # 4. Store in ChromaDB
    collection = store_in_chromadb(chunks, embeddings)

    print(f"\nChromaDB collection: {COLLECTION_NAME}")
    print(f"Stored records: {collection.count()}")

    print("\nIngestion completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()