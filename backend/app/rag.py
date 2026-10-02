import json
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

COLLECTION_NAME = "echodesk_support_policies"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
POLICIES_PATH = Path(__file__).resolve().parents[1] / "data" / "support_policies.json"
LOCAL_DB_PATH = Path(__file__).resolve().parents[2] / ".qdrant"


class SupportKnowledgeBase:
    def __init__(self) -> None:
        self.embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()

        if qdrant_url:
            client_options: dict[str, str] = {"url": qdrant_url}
            if qdrant_api_key and qdrant_api_key != "replace-me":
                client_options["api_key"] = qdrant_api_key
            self.client = QdrantClient(**client_options)
        else:
            self.client = QdrantClient(path=str(LOCAL_DB_PATH))

        self._ensure_indexed()

    def _ensure_indexed(self) -> None:
        if self.client.collection_exists(COLLECTION_NAME):
            return

        policies = json.loads(POLICIES_PATH.read_text(encoding="utf-8"))
        vectors = list(self.embedder.embed([policy["text"] for policy in policies]))
        vector_size = len(vectors[0])
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, policy["id"])),
                vector=vector.tolist(),
                payload={
                    "title": policy["title"],
                    "source": policy["source"],
                    "text": policy["text"],
                },
            )
            for policy, vector in zip(policies, vectors, strict=True)
        ]
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)

    def search(self, query: str, limit: int) -> list[dict[str, str]]:
        query_vector = next(self.embedder.embed([query])).tolist()
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "title": str(point.payload["title"]),
                "source": str(point.payload["source"]),
                "text": str(point.payload["text"]),
            }
            for point in response.points
            if point.payload is not None
        ]


_knowledge_base: SupportKnowledgeBase | None = None


def search_support_docs(query: str, limit: int = 3) -> list[dict[str, str]]:
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = SupportKnowledgeBase()
    return _knowledge_base.search(query, limit)


def close_support_docs() -> None:
    global _knowledge_base
    if _knowledge_base is not None:
        _knowledge_base.client.close()
        _knowledge_base = None