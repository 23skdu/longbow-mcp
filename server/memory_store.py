"""Vector memory storage using Longbow vector index + JSON metadata sidecar.

Longbow is a pure HNSW vector index — it stores vectors and returns integer IDs.
String columns (id, metadata) are not persisted by Longbow. We maintain a local
JSON sidecar file that maps Longbow integer indices to full Memory objects.
"""
import json
import os
import time
import uuid
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from longbow import LongbowClient
from longbow.exceptions import LongbowError, LongbowConnectionError, LongbowQueryError

from models import Memory, SearchResult

logger = logging.getLogger(__name__)

# Sidecar file path — writable directory inside Docker container
SIDECAR_PATH = os.getenv("MEMORY_SIDECAR_PATH", "/app/data/memory_sidecar.json")


def _uuid_to_graph_id(uuid_str: str) -> int:
    """Map a UUID string to an int64 graph node ID for Longbow graph operations."""
    return abs(hash(uuid_str)) % (2**53)


class MetadataSidecar:
    """Thread-safe JSON sidecar for memory metadata.

    Stores a mapping of { memory_uuid: { content, client_id, created_at, metadata, longbow_idx } }.
    Also tracks next_index to assign sequential Longbow integer IDs.
    """

    def __init__(self, path: str = SIDECAR_PATH):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {"memories": {}, "next_index": 0}
        self._load()

    def _load(self):
        """Load sidecar from disk if it exists."""
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    self._data = json.load(f)
                logger.info(f"Loaded {len(self._data.get('memories', {}))} memories from sidecar")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load sidecar ({e}), starting fresh")
                self._data = {"memories": {}, "next_index": 0}

    def _save(self):
        """Persist sidecar to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f)

    def add(self, memory_id: str, content: str, client_id: str,
            created_at: str, metadata: Dict, embedding: List[float]) -> int:
        """Add a memory and return its Longbow integer index."""
        with self._lock:
            idx = self._data["next_index"]
            self._data["memories"][memory_id] = {
                "longbow_idx": idx,
                "content": content,
                "client_id": client_id,
                "created_at": created_at,
                "metadata": metadata,
            }
            self._data["next_index"] = idx + 1
            self._save()
            return idx

    def get(self, memory_id: str) -> Optional[Dict]:
        """Get memory metadata by UUID."""
        return self._data.get("memories", {}).get(memory_id)

    def get_by_longbow_idx(self, idx: int) -> Optional[tuple[str, Dict]]:
        """Get (uuid, metadata) by Longbow integer index."""
        for mid, meta in self._data.get("memories", {}).items():
            if meta.get("longbow_idx") == idx:
                return mid, meta
        return None

    def all_memories(self) -> Dict[str, Dict]:
        """Return all memories."""
        return dict(self._data.get("memories", {}))

    def count(self) -> int:
        return len(self._data.get("memories", {}))

    def clear(self) -> int:
        """Clear all memories, return count deleted."""
        with self._lock:
            count = len(self._data.get("memories", {}))
            self._data = {"memories": {}, "next_index": 0}
            self._save()
            return count


class MemoryStore:
    """Longbow SDK-backed vector memory store with metadata sidecar."""

    NAMESPACE = "mcp_memories"
    EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

    def __init__(
        self,
        longbow_data_uri: str = None,
        longbow_meta_uri: str = None,
    ):
        self.longbow_data_uri = longbow_data_uri or os.getenv("LONGBOW_DATA_URI", "grpc://longbow:3000")
        self.longbow_meta_uri = longbow_meta_uri or os.getenv("LONGBOW_META_URI", "grpc://longbow:3001")

        self._model: Optional[SentenceTransformer] = None
        self._client: Optional[LongbowClient] = None
        self._sidecar = MetadataSidecar()
        self._initialized = False

    def _get_model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._model

    def _get_client(self) -> LongbowClient:
        """Get or create Longbow SDK client with retry logic."""
        if self._client is None:
            max_retries = 30
            delay = 2.0

            for attempt in range(max_retries):
                try:
                    client = LongbowClient(
                        uri=self.longbow_data_uri,
                        meta_uri=self.longbow_meta_uri,
                    )
                    client.connect()
                    client.list_namespaces()
                    self._client = client
                    logger.info("Connected to Longbow via SDK")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Waiting for Longbow ({attempt + 1}/{max_retries}): {e}")
                        time.sleep(delay)
                    else:
                        raise ConnectionError(f"Failed to connect to Longbow after {max_retries} attempts: {e}")

            if not self._initialized:
                try:
                    self._client.create_namespace(self.NAMESPACE)
                except Exception:
                    pass
                self._initialized = True

        return self._client

    def _sidecar_to_memory(self, memory_id: str, meta: Dict, embedding: List[float] = None) -> Memory:
        """Convert sidecar entry to Memory object."""
        return Memory(
            id=memory_id,
            content=meta.get("content", ""),
            embedding=embedding,
            metadata=meta.get("metadata", {}),
            created_at=datetime.fromisoformat(meta.get("created_at", datetime.utcnow().isoformat())),
            client_id=meta.get("client_id", "unknown"),
        )

    def add_memory(self, content: str, client_id: str, metadata: Dict = None) -> Memory:
        """Add a new memory with embedding."""
        model = self._get_model()
        embedding = model.encode(content)  # numpy float32 array

        memory_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()

        # Store metadata in sidecar, get Longbow integer index
        longbow_idx = self._sidecar.add(
            memory_id=memory_id,
            content=content,
            client_id=client_id,
            created_at=created_at,
            metadata=metadata or {},
            embedding=embedding.tolist(),
        )

        # Store vector in Longbow with integer ID
        df = pd.DataFrame({
            "id": pd.array([longbow_idx], dtype="int64"),
            "vector": [np.asarray(embedding, dtype=np.float32)],
        })

        client = self._get_client()
        client.insert(self.NAMESPACE, df)

        return Memory(
            id=memory_id,
            content=content,
            embedding=embedding.tolist(),
            metadata=metadata or {},
            created_at=datetime.fromisoformat(created_at),
            client_id=client_id,
        )

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Semantic search using vector similarity (KNN)."""
        model = self._get_model()
        query_embedding = model.encode(query).tolist()

        client = self._get_client()
        try:
            result_df = client.search(self.NAMESPACE, query_embedding, k=top_k)
        except LongbowQueryError as e:
            logger.error(f"Search failed: {e}")
            return []

        return self._search_df_to_results(result_df)

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[SearchResult]:
        """Hybrid vector+text search with alpha blending."""
        model = self._get_model()
        query_embedding = model.encode(query).tolist()

        client = self._get_client()
        try:
            result_df = client.search(
                self.NAMESPACE, query_embedding, k=top_k,
                alpha=alpha, text_query=query,
            )
        except LongbowQueryError as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

        return self._search_df_to_results(result_df)

    def search_by_id(self, memory_id: str, top_k: int = 5) -> List[SearchResult]:
        """Find memories similar to an existing memory by ID."""
        # Look up the Longbow integer index from sidecar
        meta = self._sidecar.get(memory_id)
        if not meta:
            logger.warning(f"Memory {memory_id} not found in sidecar")
            return []

        longbow_idx = meta.get("longbow_idx")
        client = self._get_client()
        try:
            raw = client.search_by_id(self.NAMESPACE, longbow_idx, k=top_k)
        except LongbowQueryError as e:
            logger.error(f"Search by ID failed: {e}")
            return []

        if not raw:
            return []

        raw_results = raw.get("results", [])
        results = []
        for item in raw_results:
            idx = int(item.get("id", -1))
            score = float(item.get("score", 0.0))
            entry = self._sidecar.get_by_longbow_idx(idx)
            if entry:
                mid, emeta = entry
                memory = self._sidecar_to_memory(mid, emeta)
                results.append(SearchResult(memory=memory, score=score))

        return results

    def filtered_search(self, query: str, top_k: int = 5, filters: List[Dict] = None) -> List[SearchResult]:
        """Vector search with metadata predicate filters."""
        model = self._get_model()
        query_embedding = model.encode(query).tolist()

        client = self._get_client()
        try:
            result_df = client.search(
                self.NAMESPACE, query_embedding, k=top_k,
                filters=filters,
            )
        except LongbowQueryError as e:
            logger.error(f"Filtered search failed: {e}")
            return []

        return self._search_df_to_results(result_df)

    def list_memories(self, limit: int = 50, offset: int = 0) -> tuple[List[Memory], int]:
        """List all memories with pagination."""
        all_mems = self._sidecar.all_memories()
        total = len(all_mems)

        # Sort by created_at descending
        sorted_items = sorted(
            all_mems.items(),
            key=lambda kv: kv[1].get("created_at", ""),
            reverse=True,
        )

        paginated = sorted_items[offset:offset + limit]
        memories = [self._sidecar_to_memory(mid, meta) for mid, meta in paginated]
        return memories, total

    def delete_all(self) -> int:
        """Delete all memories."""
        count = self._sidecar.clear()

        # Reset Longbow namespace
        client = self._get_client()
        try:
            client.delete_namespace(self.NAMESPACE)
        except Exception:
            pass
        try:
            client.create_namespace(self.NAMESPACE, force=True)
        except Exception:
            pass

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        all_mems = self._sidecar.all_memories()

        clients = set()
        oldest = None
        newest = None
        for meta in all_mems.values():
            clients.add(meta.get("client_id", "unknown"))
            created = meta.get("created_at")
            if created:
                if oldest is None or created < oldest:
                    oldest = created
                if newest is None or created > newest:
                    newest = created

        return {
            "total_memories": len(all_mems),
            "unique_clients": len(clients),
            "oldest_memory": oldest,
            "newest_memory": newest,
            "backend": "longbow",
        }

    def add_edge(self, source_id: str, target_id: str, predicate: str = "related_to", weight: float = 1.0) -> None:
        """Add a directed relationship edge between two memories."""
        client = self._get_client()
        subject = _uuid_to_graph_id(source_id)
        obj = _uuid_to_graph_id(target_id)
        client.add_edge(self.NAMESPACE, subject=subject, predicate=predicate, object=obj, weight=weight)

    def traverse(self, start_id: str, max_hops: int = 2, incoming: bool = False, decay: float = 0.0, weighted: bool = True) -> List[Dict]:
        """Graph traversal from a starting memory."""
        client = self._get_client()
        start = _uuid_to_graph_id(start_id)
        return client.traverse(
            self.NAMESPACE,
            start=start,
            max_hops=max_hops,
            incoming=incoming,
            decay=decay,
            weighted=weighted,
        )

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get Longbow dataset metadata."""
        client = self._get_client()
        try:
            return client.get_info(self.NAMESPACE)
        except Exception as e:
            logger.error(f"get_info failed: {e}")
            return {"total_records": -1, "total_bytes": -1}

    def snapshot(self) -> None:
        """Trigger manual persistence snapshot."""
        client = self._get_client()
        client.snapshot()

    # --- Internal helpers ---

    def _search_df_to_results(self, df: pd.DataFrame) -> List[SearchResult]:
        """Convert Longbow search result DataFrame to list of SearchResult.

        Longbow returns integer IDs — map them to Memory objects via sidecar.
        """
        if df is None or df.empty:
            return []

        results = []
        seen_ids: set = set()
        for _, row in df.iterrows():
            # Longbow returns uint64 IDs — convert to int
            raw_id = row.get("id")
            try:
                longbow_idx = int(float(raw_id))  # handle "0.0" string or uint64
            except (ValueError, TypeError):
                logger.warning(f"Unparseable search result ID: {raw_id}")
                continue

            score = float(row.get("score", row.get("distance", 0.0)))
            if "distance" in df.columns and "score" not in df.columns:
                score = 1.0 / (1.0 + score)

            entry = self._sidecar.get_by_longbow_idx(longbow_idx)
            if entry:
                mid, meta = entry
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)
                memory = self._sidecar_to_memory(mid, meta)
                results.append(SearchResult(memory=memory, score=score))

        return results


# Global store instance
_store: Optional[MemoryStore] = None


def get_store() -> MemoryStore:
    """Get or create global memory store."""
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store
