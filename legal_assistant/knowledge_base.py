"""Configuration, JSON/PDF loading, and Pinecone index preparation."""
import json
import os
import uuid
import hashlib
import time
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
from legal_assistant.chunking import chunk_text, token_count, CHUNKING_VERSION, MAX_TOKENS, OVERLAP_TOKENS
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "nebius").strip().lower()
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or (
    "https://api.tokenfactory.nebius.com/v1" if LLM_PROVIDER == "nebius"
    else "https://api.openai.com/v1"
)
LLM_MODEL = os.getenv("LLM_MODEL") or (
    "Qwen/Qwen3-235B-A22B-Instruct-2507" if LLM_PROVIDER == "nebius" else "gpt-4o-mini"
)
EVIDENCE_MODEL = os.getenv("EVIDENCE_MODEL") or LLM_MODEL

# Only synthetic demonstration sources are eligible until company authentication exists.


def eligible_record(record, folder):
    today = date.today().isoformat()
    if (record.get("approval_status") != "approved" or record.get("access_scope") != "demo"
            or record.get("synthetic") is not True):
        return False
    if record.get("document_status") != "current" or record.get("effective_from", "9999") > today:
        return False
    if record.get("expires_on") and record["expires_on"] <= today:
        return False
    return folder != "reviewed_matters" or (
        record.get("status") == "closed" and record.get("resolution_verified") is True
        and record.get("legal_reviewed") is True)


def read_source_records(path, folder):
    """Yield records from JSON, or extract PDF pages after checking sidecar metadata."""
    if path.name.endswith(".metadata.json"):
        return
    if path.suffix.lower() == ".json":
        for parent in json.loads(path.read_text())["records"]:
            if "sections" not in parent:
                yield {**parent, "source_file": path.name, "source_format": "json"}
                continue
            for index, section in enumerate(parent["sections"]):
                ident = section.get("id", f"{parent['id']}:s{index}")
                yield {**parent, **section, "id": ident,
                       "source_id": parent.get("source_id", parent["id"]),
                       "title": parent.get("title", ""),
                       "section_title": section.get("title", section.get("heading", "")),
                       "source_file": path.name, "source_format": "json"}
    elif path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        sidecar = path.with_suffix(".metadata.json")
        if not sidecar.exists():
            raise ValueError(f"Missing PDF metadata sidecar: {sidecar.name}")
        metadata = json.loads(sidecar.read_text())
        if not eligible_record(metadata, folder):
            return
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError(f"Encrypted PDFs are not supported: {path.name}")
        extracted = False
        for page_number, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            extracted = True
            yield {**metadata, "id": f"{metadata['id']}:p{page_number}",
                   "source_id": metadata.get("source_id", metadata["id"]),
                   "source_format": "pdf", "source_file": path.name,
                   "page": page_number, "text": text}
        if not extracted:
            raise ValueError(f"PDF contains no extractable text; OCR is not implemented: {path.name}")


def load_documents(data_dir):
    documents = []
    seen = set()
    for folder in ("legal_faqs", "policies", "reviewed_matters"):
        for path in sorted((Path(data_dir) / folder).glob("*")):
            if not path.is_file() or path.suffix.lower() not in {".json", ".pdf"}:
                continue
            for record in read_source_records(path, folder):
                if not eligible_record(record, folder):
                    continue
                ident = record["id"]
                if ident in seen:
                    raise ValueError(f"Duplicate source ID: {ident}")
                seen.add(ident)
                text = record.get("text") or record.get("content")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Missing text: {ident}")
                metadata = {k: v for k, v in record.items() if k in (
                    "source_id", "source_type", "title", "source_url", "organization",
                    "policy_version", "effective_from", "jurisdiction", "document_status", "updated_at",
                    "approval_status", "access_scope", "synthetic", "page", "source_format", "source_file")
                    and isinstance(v, (str, int, float, bool))}
                metadata.setdefault("source_id", ident)
                for index, piece in enumerate(chunk_text(text, record.get("source_type", ""),
                                                          record.get("section_title", record.get("title", "")))):
                    chunk_metadata = {**metadata,
                        "section_id": f"{ident}:s{piece['section_index']}",
                        "section_title": piece["section_title"], "chunk_index": index,
                        "chunking_version": CHUNKING_VERSION, "token_count": token_count(piece["text"]),
                        "split_fallback": piece["split_fallback"]}
                    documents.append({"id": f"{ident}:{index}", "text": piece["text"], "metadata": chunk_metadata})
    if not documents:
        raise ValueError("No current approved synthetic demo records found in the legal data folders.")
    return documents

def build_index(upload=True):
    """Publish a verified snapshot; runtime preparation only verifies, never uploads."""
    from legal_assistant.embeddings import MODEL_ID, embed_texts
    from legal_assistant.vector_store import connect_index, connection_settings
    documents = load_documents(ROOT / "data")
    name,host=connection_settings()
    digest=hashlib.sha256(json.dumps({"documents":documents,"model":MODEL_ID},sort_keys=True).encode()).hexdigest()[:20]
    namespace=os.getenv("PINECONE_NAMESPACE","legal-demo").strip()+"-"+digest
    index=connect_index(validate=True)
    if upload:
        for start in range(0,len(documents),32):
            batch=documents[start:start+32]
            embeddings=embed_texts([d['text'] for d in batch])
            index.upsert(vectors=[{"id":d['id'],"values":vector,
                         "metadata":{**d['metadata'],"text":d['text']}} for d,vector in zip(batch,embeddings)],namespace=namespace)
    # Pinecone is eventually consistent. Do not publish an unready local snapshot.
    deadline=time.monotonic()+(30 if upload else 0)
    while True:
        stats=index.describe_index_stats()
        namespace_stats=stats.get('namespaces',{}).get(namespace,{})
        count=namespace_stats.get('vector_count',namespace_stats.get('record_count',0))
        if count==len(documents):
            fetched=index.fetch(ids=[d['id'] for d in documents],namespace=namespace)
            if set(fetched.get('vectors',{}))=={d['id'] for d in documents}: break
        if time.monotonic()>=deadline:
            raise ValueError('Pinecone snapshot not ready. Run python -m legal_assistant.knowledge_base to index this dataset.')
        time.sleep(1)
    manifest={"vector_store":"pinecone","index_name":name,"host":host,"namespace":namespace,
              "embedding_model":MODEL_ID,"chunking":{"version":CHUNKING_VERSION,"max_tokens":MAX_TOKENS,"overlap_tokens":OVERLAP_TOKENS},
              "documents":documents}
    storage=ROOT/"storage"
    storage.mkdir(exist_ok=True)
    temporary=storage/f"index-{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(manifest))
    temporary.replace(storage/"index.json")
    return len(documents)

if __name__ == '__main__':
    print(f"Indexed {build_index()} passages in Pinecone.")
