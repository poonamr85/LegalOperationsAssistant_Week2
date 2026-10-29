"""One-time, idempotent MiniLM index setup. Never deletes existing indexes."""
import os
from pathlib import Path
import time

from dotenv import load_dotenv, set_key
from pinecone import Pinecone, ServerlessSpec

ROOT = Path(__file__).resolve().parents[1]
INDEX_NAME = 'legal-operations-minilm-vectors'


def create_index(client, env_path, timeout=60):
    if not client.has_index(INDEX_NAME):
        client.create_index(name=INDEX_NAME, vector_type='dense', dimension=384,
                            metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    deadline = time.monotonic() + timeout
    while True:
        info = client.describe_index(INDEX_NAME)
        if info.status.ready:
            break
        if time.monotonic() >= deadline:
            raise ValueError('Index is still provisioning; rerun this script shortly.')
        time.sleep(1)
    fields = info.schema.to_dict().get('fields', {}) if info.schema else {}
    if info.dimension != 384 or info.metric != 'cosine':
        raise ValueError('Existing index has incompatible vector settings; no configuration changed.')
    if any(f.get('embed') or f.get('sparse_embed') for f in fields.values()):
        raise ValueError('Existing index uses hosted embeddings; no configuration changed.')
    if fields and sum(f.get('type') == 'dense_vector' for f in fields.values()) != 1:
        raise ValueError('Existing index needs exactly one dense-vector field; no configuration changed.')
    host = info.host.removeprefix('https://').rstrip('/')
    for key, value in [('PINECONE_INDEX_NAME', INDEX_NAME), ('PINECONE_INDEX_HOST', host)]:
        set_key(str(env_path), key, value)
    return host


def main():
    load_dotenv(ROOT / '.env')
    key = os.getenv('PINECONE_API_KEY', '').strip()
    if not key:
        raise ValueError('Set PINECONE_API_KEY before running index setup.')
    host = create_index(Pinecone(api_key=key, timeout=20), ROOT / '.env')
    print(f'Ready: {INDEX_NAME}, 384 dimensions, cosine. Local .env updated.')
    print(f'Host: {host}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Do not dump provider payloads or credentials into shared terminal logs.
        print(f'Index setup failed ({type(exc).__name__}). Check index quota, credentials and Pinecone availability.')
        raise SystemExit(1)
