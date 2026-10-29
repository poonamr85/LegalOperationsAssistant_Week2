import json
import os
import re
from functools import lru_cache
from threading import RLock
from rank_bm25 import BM25Okapi
from legal_assistant.knowledge_base import ROOT, CHROMA_PATH

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def evidence_fingerprint(passage):
    text = passage["text"]
    # Remove only the standard FAQ question wrapper; never remove qualifiers/negation.
    if text.startswith("Question:") and "\nAnswer:" in text:
        text = text.split("\nAnswer:", 1)[1]
    meta = passage["metadata"]
    authority = "historical" if meta.get("source_type") == "reviewed_matter" else "guidance"
    return (" ".join(text.lower().split()), authority,
            *(meta.get(k) for k in ("organization", "policy_version", "jurisdiction", "effective_from", "document_status", "access_scope")))


def deduplicate_passages(passages, limit):
    unique = []
    seen = set()
    for passage in passages:
        fingerprint = evidence_fingerprint(passage)
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(passage)
            if len(unique) >= limit:
                break
    return unique


class HybridRetriever:
    def __init__(self, manifest=None, chroma_path=None):
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        manifest = manifest if manifest is not None else json.loads((ROOT / "storage/index.json").read_text())
        self.documents = manifest["documents"]
        self.collection = chromadb.PersistentClient(path=str(chroma_path or CHROMA_PATH)).get_collection(
            manifest["collection"], embedding_function=DefaultEmbeddingFunction())

        self._lock = RLock()
        self.scopes = {}
        for d in self.documents:
            meta = d["metadata"]
            if meta.get("access_scope") != "demo" or meta.get("approval_status") != "approved" or meta.get("synthetic") is not True:
                continue
            key = (meta.get("organization"), meta.get("policy_version"))
            self.scopes.setdefault(key, []).append(d)
        self.indexes = {key: BM25Okapi([tokenize(d["text"]) for d in docs]) for key, docs in self.scopes.items()}

    def search(self, question, organization, policy_version, limit=None):
        limit = int(os.getenv("RETRIEVAL_LIMIT", "4")) if limit is None else limit
        if not 1 <= limit <= 12:
            raise ValueError("RETRIEVAL_LIMIT must be between 1 and 12.")
        # Serialize lazy model initialization/use for the shared local retriever.
        with self._lock:
            return self._search(question, organization, policy_version, limit)

    def _search(self, question, organization, policy_version, limit):
        key = (organization, policy_version)
        docs = self.scopes.get(key, [])
        if not docs:
            return []
        by_id = {d["id"]: d for d in docs}
        bm25 = self.indexes[key]
        scores = bm25.get_scores(tokenize(question))
        lexical = [docs[i]["id"] for i in sorted(range(len(docs)), key=lambda i: scores[i], reverse=True) if scores[i] > 0][:12]
        result = self.collection.query(query_texts=[question], n_results=min(12, len(docs)),
            where={"$and": [{"organization": organization}, {"policy_version": policy_version}, {"access_scope": "demo"}, {"synthetic": True}, {"approval_status": "approved"}]})
        semantic = result["ids"][0]
        similarity = dict(zip(semantic, [1 - d for d in result["distances"][0]]))
        fusion = {}
        for ranking in (lexical, semantic):
            for rank, ident in enumerate(ranking, 1):
                fusion[ident] = fusion.get(ident, 0) + 1 / (60 + rank)
        ranked = [{**by_id[ident], "rrf_score": fusion[ident], "similarity": similarity.get(ident, 0)}
                  for ident in sorted(fusion, key=fusion.get, reverse=True) if ident in by_id]
        return deduplicate_passages(ranked, limit)


@lru_cache(maxsize=2)
def _cached_retriever(manifest_text, chroma_path):
    return HybridRetriever(manifest=json.loads(manifest_text), chroma_path=chroma_path)


def get_retriever():
    # An atomic manifest switch or metadata change changes the cache key.
    # No question/answer is cached, and organization filtering remains per search.
    manifest_text = (ROOT / "storage/index.json").read_text()
    return _cached_retriever(manifest_text, str(CHROMA_PATH))
