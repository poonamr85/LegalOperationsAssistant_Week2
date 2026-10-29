"""One-time resource setup safety checks; no network requests."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import tempfile
import unittest
from dotenv import dotenv_values
from scripts.create_pinecone_index import create_index, INDEX_NAME


class IndexSetupTests(unittest.TestCase):
    def client(self, exists=False, dimension=384):
        client=MagicMock()
        client.has_index.return_value=exists
        client.describe_index.return_value=SimpleNamespace(
            dimension=dimension,metric='cosine',host='https://fixture.pinecone.io',
            schema=None,status=SimpleNamespace(ready=True))
        return client

    def test_creation_sets_correct_dimensions_without_overwriting_credentials(self):
        client=self.client()
        with tempfile.TemporaryDirectory() as temp:
            env=Path(temp)/'.env';env.write_text('PINECONE_API_KEY=private-fixture\nNEBIUS_API_KEY=other-fixture\n')
            create_index(client,env)
            values=dotenv_values(env)
            self.assertEqual(values['PINECONE_API_KEY'],'private-fixture')
            self.assertEqual(values['NEBIUS_API_KEY'],'other-fixture')
            self.assertEqual(values['PINECONE_INDEX_NAME'],INDEX_NAME)
            self.assertEqual(values['PINECONE_INDEX_HOST'],'fixture.pinecone.io')
        self.assertEqual(client.create_index.call_args.kwargs['dimension'],384)
        client.delete_index.assert_not_called()

    def test_existing_compatible_index_is_reused(self):
        client=self.client(exists=True)
        with tempfile.TemporaryDirectory() as temp:create_index(client,Path(temp)/'.env')
        client.create_index.assert_not_called()
        client.delete_index.assert_not_called()

    def test_incompatible_index_leaves_configuration_unchanged(self):
        client=self.client(exists=True,dimension=1024)
        with tempfile.TemporaryDirectory() as temp:
            env=Path(temp)/'.env';env.write_text('PINECONE_INDEX_NAME=original\n')
            with self.assertRaises(ValueError):create_index(client,env)
            self.assertEqual(env.read_text(),'PINECONE_INDEX_NAME=original\n')
        client.create_index.assert_not_called()
