"""Chunking boundary and ingestion regressions using the installed MiniLM tokenizer."""
import json
import tempfile
import unittest
from pathlib import Path
from legal_assistant.chunking import chunk_text, token_count, MAX_TOKENS
from legal_assistant.knowledge_base import ROOT, load_documents


class ChunkingTests(unittest.TestCase):
    def chunks(self, text, **kwargs):
        return list(chunk_text(text, 'policy', **kwargs))

    def test_short_faq_pair_is_verbatim(self):
        text = 'Question: Can I share data?\nAnswer: Only after Privacy review and approval.'
        chunks = list(chunk_text(text, 'legal_faq'))
        self.assertEqual([p['text'] for p in chunks], [text])

    def test_long_faq_repeats_question(self):
        question = 'Question: What should I do?'
        text = question + '\nAnswer: ' + 'Keep the records and ask the Legal hold owner. ' * 30
        chunks = list(chunk_text(text, 'legal_faq', max_tokens=60, overlap_tokens=15))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(p['text'].startswith(question) for p in chunks))
        self.assertTrue(all(token_count(p['text']) <= 60 for p in chunks))

    def test_sections_keep_rule_and_restriction_together(self):
        # The old splitter lost the end of this short section at a character boundary.
        padding = 'Background information for this fictional agreement. ' * 9
        rule = 'Use an executed NDA before disclosing confidential material. An NDA alone does not authorize sharing employee data; Privacy approval is required.'
        text = '# Background\n' + padding + '\n# Employee data\n' + rule
        chunks = self.chunks(text)
        related = [p for p in chunks if p['section_title'] == 'Employee data']
        self.assertEqual(len(related), 1)
        self.assertIn(rule, related[0]['text'])
        self.assertNotIn('Background information', related[0]['text'])

    def test_oversized_section_overlaps_complete_sentences(self):
        sentences = [f'Record item {i} and preserve the required supporting documents.' for i in range(16)]
        chunks = self.chunks(' '.join(sentences), max_tokens=55, overlap_tokens=15)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(token_count(p['text']) <= 55 for p in chunks))
        self.assertTrue(all(p['text'].endswith('.') for p in chunks))
        for sentence in sentences:
            self.assertTrue(any(sentence in p['text'] for p in chunks), sentence)
        for previous, following in zip(chunks, chunks[1:]):
            last = previous['text'].split('. ')[-1]
            self.assertTrue(following['text'].startswith(last), last)
        self.assertFalse(any(p['split_fallback'] for p in chunks))

    def test_long_sentence_fallback_keeps_all_words_and_token_limit(self):
        words = [f'word{i}' for i in range(180)]
        chunks = self.chunks(' '.join(words), max_tokens=40, overlap_tokens=10)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(p['split_fallback'] for p in chunks))
        self.assertTrue(all(token_count(p['text']) <= 40 for p in chunks))
        self.assertEqual(' '.join(p['text'] for p in chunks).split(), words)

    def test_unbroken_text_and_long_heading_stay_within_budget(self):
        chunks = self.chunks('# ' + 'Heading ' * 100 + '\n' + 'abcdef' * 500,
                             max_tokens=40, overlap_tokens=10)
        self.assertTrue(chunks)
        self.assertTrue(all(token_count(p['text']) <= 40 for p in chunks))

    def test_abbreviations_and_bullets_preserve_text(self):
        chunks = self.chunks('Dr. Smith reviews it.\n\n- Preserve the NDA.\n- Ask Legal about exceptions.')
        text = '\n'.join(p['text'] for p in chunks)
        self.assertIn('Dr. Smith reviews it.', text)
        self.assertIn('- Preserve the NDA.', text)
        self.assertIn('- Ask Legal about exceptions.', text)

    def test_explicit_json_sections_inherit_source_and_have_unique_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'policies'
            folder.mkdir()
            parent = dict(id='POL-TEST', title='Test policy', source_type='policy',
                          synthetic=True, access_scope='demo', approval_status='approved',
                          document_status='current', effective_from='2000-01-01',
                          organization='TestCo', policy_version='1',
                          sections=[{'title':'Rule','text':'Preserve the documents.'},
                                    {'title':'Exception','text':'Only Legal may release a hold.'}])
            (folder/'sections.json').write_text(json.dumps({'records':[parent]}))
            docs = load_documents(tmp)
            self.assertEqual(len(docs), 2)
            self.assertEqual(len({d['id'] for d in docs}), 2)
            self.assertEqual({d['metadata']['section_title'] for d in docs}, {'Rule','Exception'})
            self.assertTrue(all(d['metadata']['source_id'] == 'POL-TEST' for d in docs))
            self.assertTrue(all(d['metadata']['organization'] == 'TestCo' for d in docs))

    def test_real_corpus_metadata_and_no_silent_token_truncation(self):
        docs = load_documents(ROOT/'data')
        self.assertEqual(len({d['id'] for d in docs}), len(docs))
        for doc in docs:
            self.assertLessEqual(token_count(doc['text']), MAX_TOKENS)
            self.assertEqual(doc['metadata']['token_count'], token_count(doc['text']))
            self.assertTrue(doc['metadata']['section_id'])
            self.assertTrue(doc['metadata']['source_file'])
        pdf = [d for d in docs if d['metadata']['source_format'] == 'pdf']
        self.assertTrue(pdf)
        self.assertTrue(all(d['metadata']['page'] == 1 for d in pdf))
        self.assertTrue(any('Historical approval' in d['metadata']['section_title'] for d in pdf))


if __name__ == '__main__':
    unittest.main()
