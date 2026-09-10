# ================================================================
# MODULE 3 - CHROMADB RETRIEVER
# ================================================================

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_support"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Load once when the module is imported.
model = SentenceTransformer(EMBEDDING_MODEL)

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_collection(name=COLLECTION_NAME)


def retrieve_documents(query: str, top_k: int = 3):
    """
    Embed the query and retrieve the top-k most similar
    documents from ChromaDB.
    """

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        # Chroma returns cosine distance.
        # Convert distance to a simple similarity score.
        similarity = 1.0 - distance

        retrieved.append(
            {
                "text": document,
                "source": metadata["source"],
                "similarity": similarity,
            }
        )

    return retrieved


if __name__ == "__main__":
    query = "How much does delivery cost?"

    results = retrieve_documents(query)

    print("=" * 60)
    print("RETRIEVAL TEST")
    print("=" * 60)

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print(f"Source: {result['source']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Text: {result['text'][:200]}...")

    print("=" * 60)