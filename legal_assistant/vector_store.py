"""Pinecone adapter for supplied dense vectors; no hosted embedding inference."""
import os


def connection_settings():
    required=['PINECONE_API_KEY','PINECONE_INDEX_NAME','PINECONE_INDEX_HOST']
    if any(not os.getenv(k,'').strip() for k in required):
        raise ValueError('Set PINECONE_API_KEY, PINECONE_INDEX_NAME and PINECONE_INDEX_HOST')
    host=os.environ['PINECONE_INDEX_HOST'].strip().removeprefix('https://').rstrip('/')
    if '/' in host or not host.endswith('.pinecone.io'):
        raise ValueError('PINECONE_INDEX_HOST must be the Pinecone index hostname')
    return os.environ['PINECONE_INDEX_NAME'].strip(),host


class VectorIndex:
    """Expose the same minimal contract for legacy vector and newer document APIs."""
    def __init__(self, index, field=None):
        self.index=index
        self.field=field

    def upsert(self, vectors, namespace):
        if not self.field:
            return self.index.upsert(vectors=vectors,namespace=namespace)
        documents=[{**v['metadata'],'_id':v['id'],self.field:v['values']} for v in vectors]
        return self.index.documents.upsert(namespace=namespace,documents=documents)

    def fetch(self, ids, namespace):
        if not self.field: return self.index.fetch(ids=ids,namespace=namespace)
        response=self.index.documents.fetch(namespace=namespace,ids=ids,include_fields=['text'])
        return {'vectors':response.documents}

    def describe_index_stats(self):
        return self.index.describe_index_stats()

    def query(self, vector, namespace, top_k, include_metadata, filter):
        if not self.field:
            return self.index.query(vector=vector,namespace=namespace,top_k=top_k,
                                    include_metadata=include_metadata,filter=filter)
        response=self.index.documents.search(namespace=namespace,top_k=top_k,filter=filter,
            score_by=[{'type':'dense_vector','fields':[self.field],'values':vector}])
        return {'matches':[{'id':m.id,'score':m.score} for m in response.matches]}


def connect_index(validate=False):
    from pinecone import Pinecone
    name,host=connection_settings()
    client=Pinecone(api_key=os.environ['PINECONE_API_KEY'],timeout=20)
    info=client.describe_index(name)
    if info.dimension != 384 or info.metric != 'cosine':
        raise ValueError('Pinecone index must use 384 dimensions and cosine similarity')
    if info.host.removeprefix('https://').rstrip('/') != host or not info.status.ready:
        raise ValueError('Pinecone host does not match a ready configured index')
    schema=info.schema.to_dict() if info.schema is not None else {}
    fields=schema.get('fields',{})
    if any(f.get('embed') or f.get('sparse_embed') for f in fields.values()):
        raise ValueError('This Pinecone index uses hosted embeddings. Create a bring-your-own dense-vector index for MiniLM.')
    dense=[n for n,f in fields.items() if f.get('type')=='dense_vector']
    if fields and len(dense)!=1:
        raise ValueError('Pinecone schema must contain exactly one dense-vector field')
    # Pinecone describes legacy vectors with reserved _values/_sparse_values fields.
    # These are a compatibility schema, not a documents-API index.
    field=dense[0] if dense and dense[0] != '_values' else None
    return VectorIndex(client.Index(host=host),field=field)
