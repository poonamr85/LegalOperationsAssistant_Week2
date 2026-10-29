"""Offline storage/embedding contracts; no external services called."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from legal_assistant import knowledge_base as kb
from legal_assistant import vector_store
from legal_assistant.embeddings import embed_texts, MODEL_ID
from tests.test_bot import PASSAGES


class PineconeMigrationTests(unittest.TestCase):
    def index(self):
        index=MagicMock()
        index.describe_index_stats.side_effect=lambda: {'namespaces':{self.namespace:{'vector_count':len(PASSAGES)}}}
        index.fetch.return_value={'vectors':{d['id']:{} for d in PASSAGES}}
        return index

    def run_build(self,upload=True,index=None):
        import hashlib
        digest=hashlib.sha256(json.dumps({'documents':PASSAGES,'model':MODEL_ID},sort_keys=True).encode()).hexdigest()[:20]
        self.namespace='fixture-'+digest
        index=index or self.index()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(kb,'ROOT',root),patch.object(kb,'load_documents',return_value=PASSAGES), \
                 patch('legal_assistant.vector_store.connection_settings',return_value=('fixture','fixture.pinecone.io')), \
                 patch('legal_assistant.vector_store.connect_index',return_value=index), \
                 patch('legal_assistant.embeddings.embed_texts',return_value=[[0.0]*384]), \
                 patch.dict('os.environ',{'PINECONE_NAMESPACE':'fixture'}):
                kb.build_index(upload=upload)
            manifest=json.loads((root/'storage/index.json').read_text())
        return manifest,index

    def test_local_embeddings_are_normalized_and_share_semantic_space(self):
        vectors=np.array(embed_texts(['Vendor NDA review','Vendor NDA review','Delete old records']))
        self.assertEqual(vectors.shape,(3,384))
        np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
        np.testing.assert_allclose(vectors[0],vectors[1],atol=1e-6)
        self.assertGreater(float(vectors[0]@vectors[1]),float(vectors[0]@vectors[2]))

    def test_upload_preserves_citations_text_and_metadata(self):
        manifest,index=self.run_build()
        vector=index.upsert.call_args.kwargs['vectors'][0]
        self.assertEqual(vector['id'],PASSAGES[0]['id'])
        self.assertEqual(vector['metadata']['text'],PASSAGES[0]['text'])
        self.assertEqual(manifest['namespace'],self.namespace)
        self.assertNotIn('api_key',json.dumps(manifest))

    def test_startup_verifies_without_uploading(self):
        _,index=self.run_build(upload=False)
        index.upsert.assert_not_called()
        index.fetch.assert_called_once()

    def test_failed_readiness_preserves_previous_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'storage').mkdir()
            manifest=root/'storage/index.json';manifest.write_text('previous snapshot')
            index=MagicMock();index.describe_index_stats.return_value={'namespaces':{}}
            with patch.object(kb,'ROOT',root),patch.object(kb,'load_documents',return_value=PASSAGES), \
                 patch('legal_assistant.vector_store.connection_settings',return_value=('fixture','fixture.pinecone.io')), \
                 patch('legal_assistant.vector_store.connect_index',return_value=index):
                with self.assertRaises(ValueError):kb.build_index(upload=False)
            self.assertEqual(manifest.read_text(),'previous snapshot')

    def test_wrong_dimensions_rejected_before_data_operations(self):
        client=MagicMock()
        client.describe_index.return_value=SimpleNamespace(dimension=1024,metric='cosine')
        with patch('pinecone.Pinecone',return_value=client),patch.dict('os.environ',{
            'PINECONE_API_KEY':'fixture','PINECONE_INDEX_NAME':'fixture','PINECONE_INDEX_HOST':'fixture.pinecone.io'}):
            with self.assertRaises(ValueError):vector_store.connect_index(validate=True)
        client.Index.assert_not_called()

    def test_retrieval_filters_namespace_and_discards_unknown_ids(self):
        from legal_assistant.retriever import HybridRetriever
        index=MagicMock();index.query.return_value={'matches':[{'id':'unknown','score':1},{'id':PASSAGES[0]['id'],'score':.8}]}
        manifest={'vector_store':'pinecone','embedding_model':MODEL_ID,'index_name':'fixture','host':'fixture.pinecone.io',
                  'namespace':'snapshot','documents':[{**d,'metadata':{**d['metadata'],'access_scope':'demo','synthetic':True,'approval_status':'approved'}} for d in PASSAGES]}
        with patch('legal_assistant.retriever.connection_settings',return_value=('fixture','fixture.pinecone.io')), \
             patch('legal_assistant.retriever.embed_texts',return_value=[[0.0]*384]):
            r=HybridRetriever(manifest,index=index)
            result=r.search('legal intake','ExampleCo (fictional)','2026.1')
            self.assertEqual([p['id'] for p in result],[PASSAGES[0]['id']])
            args=index.query.call_args.kwargs
            self.assertEqual(args['namespace'],'snapshot')
            self.assertIn({'organization':{'$eq':'ExampleCo (fictional)'}},args['filter']['$and'])
            self.assertEqual(r.search('legal','Other','2026.1'),[])

    def test_hosted_embedding_index_is_rejected_without_writes(self):
        client=MagicMock()
        client.describe_index.return_value=SimpleNamespace(
            dimension=384,metric='cosine',host='https://fixture.pinecone.io',status=SimpleNamespace(ready=True),
            schema=SimpleNamespace(to_dict=lambda:{'fields':{'text':{'type':'string','embed':{'model':'llama'}}}}))
        with patch('pinecone.Pinecone',return_value=client),patch.dict('os.environ',{
            'PINECONE_API_KEY':'fixture','PINECONE_INDEX_NAME':'fixture','PINECONE_INDEX_HOST':'fixture.pinecone.io'}):
            with self.assertRaisesRegex(ValueError,'hosted embeddings'):vector_store.connect_index()
        client.Index.assert_not_called()

    def test_reserved_legacy_schema_uses_vector_api(self):
        client=MagicMock()
        client.describe_index.return_value=SimpleNamespace(
            dimension=384,metric='cosine',host='https://fixture.pinecone.io',status=SimpleNamespace(ready=True),
            schema=SimpleNamespace(to_dict=lambda:{'fields':{'_values':{'type':'dense_vector'},'_sparse_values':{'type':'sparse_vector'}}}))
        with patch('pinecone.Pinecone',return_value=client),patch.dict('os.environ',{
            'PINECONE_API_KEY':'fixture','PINECONE_INDEX_NAME':'fixture','PINECONE_INDEX_HOST':'fixture.pinecone.io'}):
            adapter=vector_store.connect_index()
        adapter.upsert([{'id':'id','values':[0.0]*384,'metadata':{}}],'snapshot')
        client.Index.return_value.upsert.assert_called_once()
        client.Index.return_value.documents.upsert.assert_not_called()

    def test_document_schema_adapter_preserves_vectors_ids_and_filters(self):
        raw=MagicMock()
        raw.documents.fetch.return_value=SimpleNamespace(documents={'id':{}})
        raw.documents.search.return_value=SimpleNamespace(matches=[SimpleNamespace(id='id',score=.9)])
        adapter=vector_store.VectorIndex(raw,field='embedding')
        adapter.upsert([{'id':'id','values':[0.0]*384,'metadata':{'text':'sample'}}],'snapshot')
        doc=raw.documents.upsert.call_args.kwargs['documents'][0]
        self.assertEqual(doc['_id'],'id')
        self.assertEqual(len(doc['embedding']),384)
        self.assertEqual(set(adapter.fetch(['id'],'snapshot')['vectors']),{'id'})
        result=adapter.query([0.0]*384,'snapshot',4,False,{'organization':{'$eq':'ExampleCo'}})
        self.assertEqual(result['matches'][0]['id'],'id')
        self.assertEqual(raw.documents.search.call_args.kwargs['filter'],{'organization':{'$eq':'ExampleCo'}})

    def test_snapshot_changes_for_changed_documents(self):
        first,_=self.run_build()
        with patch.dict(PASSAGES[0],text='Changed policy'):
            second,_=self.run_build()
        self.assertNotEqual(first['namespace'],second['namespace'])

    def test_startup_stops_if_snapshot_verification_fails(self):
        from scripts import start_app
        with patch('scripts.prepare_app.main',side_effect=ValueError('unready')),patch('os.execv') as launch:
            with self.assertRaises(ValueError):start_app.main()
            launch.assert_not_called()
