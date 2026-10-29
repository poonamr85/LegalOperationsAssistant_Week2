import json
import os
import re
from functools import lru_cache
from threading import RLock
from rank_bm25 import BM25Okapi
from legal_assistant.knowledge_base import ROOT
from legal_assistant.embeddings import embed_texts, MODEL_ID
from legal_assistant.vector_store import connect_index, connection_settings

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
    def __init__(self, manifest=None, index=None):
        manifest = manifest if manifest is not None else json.loads((ROOT / "storage/index.json").read_text())
        if manifest.get("vector_store") != "pinecone" or manifest.get("embedding_model") != MODEL_ID:
            raise ValueError("Prepare the Pinecone index before searching")
        name,host=connection_settings()
        if (name,host)!=(manifest['index_name'],manifest['host']):
            raise ValueError("Pinecone configuration differs from the prepared index")
        self.documents = manifest["documents"]
        self.index = index if index is not None else connect_index()
        self.namespace = manifest['namespace']

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
        result = self.index.query(vector=embed_texts([question])[0], namespace=self.namespace,
            top_k=min(12,len(docs)), include_metadata=False,
            filter={"$and": [{"organization": {"$eq":organization}}, {"policy_version": {"$eq":policy_version}},
                    {"access_scope":{"$eq":"demo"}}, {"synthetic":{"$eq":True}}, {"approval_status":{"$eq":"approved"}}]})
        semantic=[match['id'] for match in result['matches'] if match['id'] in by_id]
        similarity={match['id']:match['score'] for match in result['matches']}
        fusion = {}
        for ranking in (lexical, semantic):
            for rank, ident in enumerate(ranking, 1):
                fusion[ident] = fusion.get(ident, 0) + 1 / (60 + rank)
        ranked = [{**by_id[ident], "rrf_score": fusion[ident], "similarity": similarity.get(ident, 0)}
                  for ident in sorted(fusion, key=fusion.get, reverse=True) if ident in by_id]
        return deduplicate_passages(ranked, limit)


@lru_cache(maxsize=2)
def _cached_retriever(manifest_text, credential):
    return HybridRetriever(manifest=json.loads(manifest_text))


def get_retriever():
    # An atomic manifest switch or metadata change changes the cache key.
    # No question/answer is cached, and organization filtering remains per search.
    manifest_text = (ROOT / "storage/index.json").read_text()
    return _cached_retriever(manifest_text, os.getenv("PINECONE_API_KEY", ""))
