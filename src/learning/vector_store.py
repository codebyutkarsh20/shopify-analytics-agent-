"""SQLite-based vector store for semantic search.

Provides embedding storage and cosine-similarity retrieval for templates,
error lessons, and recovery patterns. Uses sentence-transformers with lazy
model loading so bot startup is not affected.

Falls back gracefully to empty results if the embedding model is
unavailable — the rest of the system continues with keyword matching.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Dimensionality of all-MiniLM-L6-v2 embeddings
EMBEDDING_DIM = 384


class EmbeddingStore:
    """Manages embeddings for semantic search of templates and learnings.

    Stores embedding vectors as numpy float32 BLOBs in an ``embedding_vectors``
    SQLite table and performs brute-force cosine-similarity search at query time.
    For typical template counts (< 1 000) this is faster than any index.
    """

    def __init__(
        self,
        db_ops,
        model_name: str = "all-MiniLM-L6-v2",
        lazy_load: bool = True,
    ):
        """Initialise the embedding store.

        Args:
            db_ops: DatabaseOperations instance for persistence.
            model_name: HuggingFace model identifier for sentence-transformers.
            lazy_load: If True the model is loaded on first ``embed()`` call,
                       keeping bot startup fast.
        """
        self.db_ops = db_ops
        self.model_name = model_name
        self._model = None
        self._model_load_attempted = False
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._available = True  # False after a failed load attempt

        if not lazy_load:
            self._ensure_model_loaded()

        logger.info(
            "EmbeddingStore initialised",
            model=model_name,
            lazy_load=lazy_load,
        )

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> bool:
        """Lazy-load the sentence-transformers model.

        Returns True if the model is ready, False otherwise.  After one
        failed attempt the method short-circuits so we don't retry on
        every request.
        """
        if self._model is not None:
            return True

        if self._model_load_attempted:
            return False

        self._model_load_attempted = True

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded: %s", self.model_name)
            return True
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — vector search disabled. "
                "Install with: pip install sentence-transformers"
            )
            self._available = False
            return False
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            self._available = False
            return False

    @property
    def is_available(self) -> bool:
        """Whether the embedding model is loaded or loadable."""
        if self._model is not None:
            return True
        if self._model_load_attempted:
            return self._available
        # Haven't tried yet — assume available until proven otherwise
        return True

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def embed(self, text: str) -> Optional[np.ndarray]:
        """Embed a single text string into a 384-dim float32 vector.

        Returns None on any failure (model not loaded, empty text, etc.).
        """
        if not text or not isinstance(text, str):
            return None

        text = text.strip()
        if not text:
            return None

        # Cache hit
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        if not self._ensure_model_loaded():
            return None

        try:
            vector = self._model.encode(text, convert_to_numpy=True).astype(np.float32)
            self._embedding_cache[text] = vector
            return vector
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Embed multiple texts efficiently in one model call."""
        if not self._ensure_model_loaded():
            return [None] * len(texts)

        # Split cached / uncached
        uncached_texts = []
        uncached_indices = []
        for i, t in enumerate(texts):
            t_stripped = (t or "").strip()
            if t_stripped and t_stripped not in self._embedding_cache:
                uncached_texts.append(t_stripped)
                uncached_indices.append(i)

        # Batch encode uncached
        if uncached_texts:
            try:
                vectors = self._model.encode(uncached_texts, convert_to_numpy=True).astype(np.float32)
                for t, vec in zip(uncached_texts, vectors):
                    self._embedding_cache[t] = vec
            except Exception as exc:
                logger.warning("Batch embedding failed: %s", exc)

        # Assemble results
        results: List[Optional[np.ndarray]] = []
        for t in texts:
            t_stripped = (t or "").strip()
            results.append(self._embedding_cache.get(t_stripped))
        return results

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store_embedding(
        self,
        entity_type: str,
        entity_id: int,
        text: str,
    ) -> bool:
        """Compute embedding for *text* and persist it.

        Args:
            entity_type: ``"template"`` | ``"error_lesson"``
            entity_id:   Primary key in the source table.
            text:        Text to embed (description + examples).

        Returns True on success, False on failure (non-fatal).
        """
        vector = self.embed(text)
        if vector is None:
            return False

        try:
            self.db_ops.upsert_embedding(
                entity_type=entity_type,
                entity_id=entity_id,
                embedding_bytes=vector.tobytes(),
                source_text=text[:500],
            )
            return True
        except Exception as exc:
            logger.warning(
                "Failed to store embedding %s:%d — %s",
                entity_type,
                entity_id,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_similar(
        self,
        query_text: str,
        entity_type: str,
        top_k: int = 5,
        min_similarity: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """Find entities whose embeddings are most similar to *query_text*.

        Args:
            query_text:     The user query or search phrase.
            entity_type:    ``"template"`` or ``"error_lesson"``.
            top_k:          Maximum results to return.
            min_similarity: Minimum cosine similarity threshold (0–1).

        Returns:
            List of dicts sorted by descending similarity::

                [{"entity_id": 3, "similarity": 0.82, "source_text": "..."},
                 ...]

            Returns ``[]`` on any failure so the caller can fall back to
            keyword matching.
        """
        query_vec = self.embed(query_text)
        if query_vec is None:
            return []

        try:
            rows = self.db_ops.get_all_embeddings(entity_type)
        except Exception as exc:
            logger.warning("Failed to load embeddings for search: %s", exc)
            return []

        if not rows:
            return []

        # Vectorised cosine similarity
        # rows is List[(entity_id, numpy_array, source_text)]
        entity_ids = []
        source_texts = []
        matrix_rows = []
        for eid, vec, stxt in rows:
            if vec is not None and vec.shape == query_vec.shape:
                entity_ids.append(eid)
                source_texts.append(stxt)
                matrix_rows.append(vec)

        if not matrix_rows:
            return []

        matrix = np.stack(matrix_rows)  # (N, 384)
        # Normalise
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        matrix_norm = matrix / norms

        similarities = matrix_norm @ query_norm  # (N,)

        # Filter and sort
        scored = []
        for idx, sim in enumerate(similarities):
            if sim >= min_similarity:
                scored.append(
                    {
                        "entity_id": entity_ids[idx],
                        "similarity": float(sim),
                        "source_text": source_texts[idx],
                    }
                )

        scored.sort(key=lambda x: x["similarity"], reverse=True)

        logger.debug(
            "Vector search: %d candidates, %d above threshold %.2f",
            len(entity_ids),
            len(scored),
            min_similarity,
        )

        return scored[:top_k]

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------

    def backfill_templates(self) -> int:
        """Embed all existing templates that lack an embedding.

        Safe to call repeatedly — skips templates that already have an
        entry in ``embedding_vectors``.

        Returns the number of newly embedded templates.
        """
        if not self._ensure_model_loaded():
            logger.warning("Skipping backfill — embedding model not available")
            return 0

        try:
            from src.database.models import QueryTemplate
            session = self.db_ops.get_session()
            try:
                templates = session.query(QueryTemplate).all()
            finally:
                session.close()
        except Exception as exc:
            logger.error("Failed to load templates for backfill: %s", exc)
            return 0

        # Get IDs that already have embeddings
        try:
            existing = self.db_ops.get_all_embeddings("template")
            existing_ids = {eid for eid, _, _ in existing}
        except Exception:
            existing_ids = set()

        count = 0
        for tpl in templates:
            if tpl.id in existing_ids:
                continue

            embed_text = self._build_template_embed_text(tpl)
            if self.store_embedding("template", tpl.id, embed_text):
                count += 1

        if count:
            logger.info("Backfilled embeddings for %d templates", count)
        else:
            logger.info("No templates needed embedding backfill")

        return count

    @staticmethod
    def _build_template_embed_text(template) -> str:
        """Build the text to embed for a QueryTemplate row.

        Combines the description with stored example queries for a richer
        semantic signal.
        """
        parts = []
        if template.intent_description:
            parts.append(template.intent_description)

        if template.example_queries:
            try:
                examples = json.loads(template.example_queries)
                if isinstance(examples, list):
                    parts.append("Examples: " + ", ".join(str(e) for e in examples))
            except (json.JSONDecodeError, TypeError):
                pass

        return ". ".join(parts) if parts else (template.intent_category or "")
