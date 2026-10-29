import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from pydantic import ValidationError
from legal_assistant.knowledge_base import ROOT, load_documents
from legal_assistant.bot import (Draft, EvidenceGrade, ClaimAssessment, ClaimValidation, create_client, grade_passes,
                 citations_valid, requests_human, pipeline_error_message)

PASSAGES = [{'id': 'LEGAL-FAQ-002:0', 'text': 'Submit the agreement through the legal intake form.', 'metadata': {'organization': 'ExampleCo (fictional)', 'policy_version': '2026.1', 'source_type': 'legal_faq', 'title': 'Contract review intake'}}]

def grade(**changes):
    return EvidenceGrade(**{**dict(relevant=True, evidence_support_percent=90, coverage="full", unanswered_parts=[], consistent=True,
                                   selected_ids=['LEGAL-FAQ-002:0'], policy_applicability="applicable", applicability_reason="Matching fictional organization and current policy edition.", human_judgment_required=False, human_judgment_reason="Documented intake guidance only.", reason='Contract intake is documented.'), **changes})

def supported_validation():
    return ClaimValidation(all_claims_assessed=True, requires_human_decision=False,
        claims=[ClaimAssessment(unit_id=0, claim='Use legal intake', verdict='supported',
                evidence_ids=['LEGAL-FAQ-002:0'], reason='The evidence describes legal intake.')],
        nonfactual_unit_ids=[])

def result(value):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=value, refusal=None))])

class SupportTests(unittest.TestCase):
    def test_sample_sources(self):
        docs = load_documents(ROOT/'data')
        json_docs = [d for d in docs if d["metadata"].get("source_format") != "pdf"]
        self.assertEqual(len({d['metadata']['source_id'] for d in json_docs}), 43)
        self.assertTrue(any(d["metadata"].get("source_id") == "POL-PDF-001" for d in docs))
        self.assertTrue(all(d["metadata"]["synthetic"] is True and d["metadata"]["access_scope"] == "demo" for d in docs))
        self.assertEqual({d['metadata']['source_type'] for d in docs}, {'legal_faq', 'policy', 'reviewed_matter'})

    def test_restricted_and_unverified_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ('legal_faqs', 'policies', 'reviewed_matters'): (root/folder).mkdir()
            r = dict(id='ok', text='Demo answer', approval_status='approved', access_scope='demo', synthetic=True, document_status='current', effective_from='2000-01-01')
            (root/'legal_faqs/a.json').write_text(json.dumps({'records':[r, {**r,'id':'private','access_scope':'internal'}]}))
            (root/'reviewed_matters/a.json').write_text(json.dumps({'records':[{**r,'id':'ticket','status':'closed','resolution_verified':False}]}))
            self.assertEqual([d['id'] for d in load_documents(root)], ['ok:0'])
            # Missing synthetic marker, superseded, future and expired records stay out.
            excluded = [{**r, 'id':'real', 'synthetic':False},
                        {**r, 'id':'old', 'document_status':'superseded'},
                        {**r, 'id':'future', 'effective_from':'9999-01-01'},
                        {**r, 'id':'expired', 'expires_on':'2001-01-01'}]
            (root/'legal_faqs/a.json').write_text(json.dumps({'records':[r]+excluded}))
            self.assertEqual([d['id'] for d in load_documents(root)], ['ok:0'])

    def test_pdf_text_and_page_metadata(self):
        docs = load_documents(ROOT/'data')
        pdf = [d for d in docs if d['metadata'].get('source_format') == 'pdf']
        self.assertTrue(pdf)
        self.assertTrue(all(d['metadata']['page'] == 1 for d in pdf))
        self.assertTrue(all(d['metadata']['source_file'] == 'fictional_ai_tools_policy.pdf' for d in pdf))
        self.assertTrue(all(d['id'].startswith('POL-PDF-001:p1:') for d in pdf))
        self.assertIn('Draft and unsigned agreements', ' '.join(d['text'] for d in pdf))

    def test_pdf_requires_eligible_sidecar(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'policies'
            folder.mkdir()
            target = folder/'sample.pdf'
            shutil.copyfile(ROOT/'data/policies/fictional_ai_tools_policy.pdf', target)
            with self.assertRaisesRegex(ValueError, 'Missing PDF metadata'):
                load_documents(tmp)
            target.with_suffix('.metadata.json').write_text(json.dumps({'synthetic':False}))
            with self.assertRaisesRegex(ValueError, 'No current approved'):
                load_documents(tmp)

    def test_valid_and_invented_citations(self):
        draft=Draft(action='answer', answer='Use legal intake [LEGAL-FAQ-002:0]', citation_ids=['LEGAL-FAQ-002:0'])
        self.assertTrue(citations_valid(draft,PASSAGES))
        draft.answer='Use legal intake [FAKE]'
        self.assertFalse(citations_valid(draft,PASSAGES))

    def test_grade_requires_all_checks_and_real_ids(self):
        self.assertTrue(grade_passes(grade(),PASSAGES))
        for changes in ({'relevant':False},{'coverage':'none'},{'consistent':False},{'selected_ids':[]},{'selected_ids':['FAKE']}):
            with self.subTest(changes=changes):
                self.assertFalse(grade_passes(grade(**changes),PASSAGES))

    def test_partial_coverage_is_accepted(self):
        self.assertTrue(grade_passes(grade(coverage="partial", unanswered_parts=["Processing time"]), PASSAGES))

    def test_support_percentage_is_bounded_and_strict(self):
        for invalid in (-1, 101, "90", 85.5):
            with self.subTest(value=invalid):
                with self.assertRaises(ValidationError): grade(evidence_support_percent=invalid)
        self.assertEqual(grade(evidence_support_percent=0).evidence_support_percent, 0)
        self.assertEqual(grade(evidence_support_percent=100).evidence_support_percent, 100)
        self.assertFalse(grade_passes(grade(evidence_support_percent=100, consistent=False), PASSAGES))

    def test_grade_rejects_malformed_boolean(self):
        with self.assertRaises(ValidationError): grade(relevant='yes')

class ModelFlowTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict('os.environ', {'LLM_PROVIDER':'nebius','NEBIUS_API_KEY':'test-only'})
        self.env.start()
        self.addCleanup(self.env.stop)
        from legal_assistant.bot import clear_client_cache
        clear_client_cache()
        self.addCleanup(clear_client_cache)
        self.patch=patch('openai.OpenAI')
        self.client=self.patch.start()
        self.addCleanup(self.patch.stop)
        self.client.return_value.with_options.return_value=self.client.return_value
        self.parse=self.client.return_value.chat.completions.parse

    def run_request(self, question='How do I request contract review?', passages=None):
        from legal_assistant.workflow import run_workflow
        with patch('legal_assistant.workflow.get_retriever') as retriever:
            retriever.return_value.search.return_value = PASSAGES if passages is None else passages
            return run_workflow(question, 'ExampleCo (fictional)', '2026.1')

    def test_grade_generate_then_one_claim_check(self):
        draft=Draft(action='answer',answer='Use legal intake [LEGAL-FAQ-002:0]',citation_ids=['LEGAL-FAQ-002:0'])
        self.parse.side_effect=[result(grade()),result(draft),result(supported_validation())]
        outcome=self.run_request()
        self.assertEqual(outcome['draft'],draft)
        self.assertEqual(outcome['grade'],grade())
        self.assertEqual(outcome['action'],'answer')
        self.assertEqual(self.parse.call_count,3)
        self.assertIs(self.parse.call_args_list[2].kwargs['response_format'],ClaimValidation)
        self.assertIs(self.parse.call_args_list[0].kwargs['response_format'],EvidenceGrade)
        self.assertIs(self.parse.call_args_list[1].kwargs['response_format'],Draft)
        self.assertEqual(self.client.call_args.kwargs['api_key'],'test-only')
        self.assertEqual(self.client.call_args.kwargs['base_url'],'https://api.tokenfactory.nebius.com/v1')

    def test_insufficient_irrelevant_conflicting_or_invalid_evidence_never_generates(self):
        for changes in ({'coverage':'none'},{'relevant':False},{'consistent':False},{'selected_ids':['FAKE']}):
            with self.subTest(changes=changes):
                self.parse.reset_mock()
                self.parse.return_value=result(grade(**changes))
                self.assertEqual(self.run_request()['action'],'escalate')
                self.assertEqual(self.parse.call_count,1)

    def test_grader_refusal_or_failure_never_generates(self):
        self.parse.return_value=result(None)
        outcome=self.run_request()
        self.assertEqual(outcome['action'],'escalate')
        self.assertEqual(outcome['failed_stage'],'evidence grading')
        self.assertIsNone(outcome['grade'])
        self.assertEqual(self.parse.call_count,1)
        self.parse.reset_mock()
        self.parse.side_effect=TimeoutError('private failure')
        outcome=self.run_request()
        self.assertEqual(outcome['action'],'escalate')
        self.assertNotIn('private failure',str(outcome))
        self.assertEqual(self.parse.call_count,1)

    def test_empty_evidence_and_human_request_bypass_model(self):
        self.assertTrue(requests_human('I want a human'))
        self.assertEqual(self.run_request(passages=[])['action'],'escalate')
        self.assertEqual(self.run_request('I want a human')['action'],'escalate')
        self.client.assert_not_called()

    def test_only_graded_passages_reach_answer_model(self):
        extra={'id':'irrelevant:0','text':'Unrelated text'}
        draft=Draft(action='answer',answer='Use legal intake [LEGAL-FAQ-002:0]',citation_ids=['LEGAL-FAQ-002:0'])
        self.parse.side_effect=[result(grade()),result(draft),result(supported_validation())]
        self.run_request(passages=PASSAGES+[extra])
        payload=json.loads(self.parse.call_args_list[1].kwargs['messages'][1]['content'])
        self.assertEqual(payload['evidence'],PASSAGES)

    def test_answer_cannot_cite_excluded_evidence(self):
        draft=Draft(action='answer',answer='Answer [excluded:0]',citation_ids=['excluded:0'])
        self.parse.side_effect=[result(grade()),result(draft),result(supported_validation())]
        outcome=self.run_request(passages=PASSAGES+[{'id':'excluded:0','text':'other'}])
        self.assertEqual(outcome['action'],'escalate')

    def test_answer_can_still_clarify(self):
        draft=Draft(action='clarify',answer='Which file type?',citation_ids=[])
        self.parse.side_effect=[result(grade()),result(draft),result(supported_validation())]
        self.assertEqual(self.run_request()['action'],'clarify')

    def test_answer_refusal_fails_closed(self):
        self.parse.side_effect=[result(grade()),result(None)]
        outcome=self.run_request()
        self.assertEqual(outcome['action'],'escalate')
        self.assertEqual(outcome['failed_stage'],'answer generation')
        self.assertEqual(self.parse.call_count,2)

class ProviderTests(unittest.TestCase):
    def setUp(self):
        from legal_assistant.bot import clear_client_cache
        clear_client_cache()
        self.addCleanup(clear_client_cache)

    def test_nebius_does_not_use_openai_key(self):
        with patch.dict('os.environ',{'LLM_PROVIDER':'nebius','OPENAI_API_KEY':'wrong'},clear=True),patch('openai.OpenAI') as client:
            with self.assertRaisesRegex(ValueError,'NEBIUS_API_KEY'): create_client()
            client.assert_not_called()

    def test_no_key_does_not_call_api(self):
        with patch.dict('os.environ',{},clear=True),patch('openai.OpenAI') as client:
            with self.assertRaises(ValueError): create_client()
            client.assert_not_called()

    def test_openai_explicit_selection(self):
        with patch.dict('os.environ',{'LLM_PROVIDER':'openai','OPENAI_API_KEY':'test'},clear=True),patch('openai.OpenAI') as client:
            client.return_value.chat.completions.parse.return_value=result(grade(coverage="none"))
            create_client()
            self.assertEqual(client.call_args.kwargs['base_url'],'https://api.openai.com/v1')

    def test_errors_do_not_expose_secrets(self):
        secret='SECRET-MUST-NOT-APPEAR'
        self.assertNotIn(secret,pipeline_error_message(ValueError(secret)))
        self.assertIn('Document search failed',pipeline_error_message(ValueError(secret),'search'))
        self.assertIn('credits are exhausted',pipeline_error_message(SimpleNamespace(code='credit_balance_exhausted')))

class GraphTests(unittest.TestCase):
    def setUp(self):
        from legal_assistant import workflow
        self.workflow=workflow
        self.search_patch=patch('legal_assistant.workflow.get_retriever')
        self.search=self.search_patch.start().return_value.search
        self.addCleanup(self.search_patch.stop)
        self.search.return_value=PASSAGES
        self.grade_patch=patch('legal_assistant.bot.grade_evidence',return_value=grade())
        self.grader=self.grade_patch.start()
        self.addCleanup(self.grade_patch.stop)
        self.answer_patch=patch('legal_assistant.bot.generate_answer',return_value=Draft(action='answer',answer='Use intake [LEGAL-FAQ-002:0]',citation_ids=['LEGAL-FAQ-002:0']))
        self.generator=self.answer_patch.start()
        self.addCleanup(self.answer_patch.stop)
        self.validator_patch=patch('legal_assistant.bot.validate_claim_support',return_value=supported_validation())
        self.validator=self.validator_patch.start()
        self.addCleanup(self.validator_patch.stop)

    def run_request(self,question='How do I request contract review?'):
        return self.workflow.run_workflow(question,'ExampleCo (fictional)','2026.1')

    def test_answer_path_and_scope(self):
        outcome=self.run_request()
        self.assertEqual(outcome['steps'],['retrieve','grade','generate','validate','check_claims'])
        self.assertEqual(outcome['action'],'answer')
        self.search.assert_called_once_with('How do I request contract review?','ExampleCo (fictional)','2026.1')
        self.assertEqual(outcome['selected_passages'],PASSAGES)

    def test_partial_evidence_still_answers(self):
        self.grader.return_value=grade(coverage='partial',unanswered_parts=['Review timing'])
        self.assertEqual(self.run_request()['action'],'answer')

    def test_rejected_grades_skip_generation(self):
        for changes in ({'coverage':'none'},{'consistent':False},{'selected_ids':['FAKE']}):
            with self.subTest(changes=changes):
                self.grader.return_value=grade(**changes)
                self.generator.reset_mock()
                outcome=self.run_request()
                self.assertEqual(outcome['steps'],['retrieve','grade','escalate'])
                self.assertEqual(outcome['action'],'escalate')
                self.generator.assert_not_called()

    def test_missing_scope_returns_setup_message_without_search_or_models(self):
        outcome=self.workflow.run_workflow('How do I request review?', '', '')
        self.assertEqual(outcome['action'],'index_needed')
        self.assertIn('administrator',outcome['answer'])
        self.search.assert_not_called()
        self.grader.assert_not_called()
        self.generator.assert_not_called()

    def test_human_request_without_scope_still_routes_to_legal(self):
        outcome=self.workflow.run_workflow('Please contact the Legal team', '', '')
        self.assertEqual(outcome['action'],'escalate')
        self.assertEqual(outcome['steps'],['escalate'])
        self.search.assert_not_called()
        self.grader.assert_not_called()
        self.generator.assert_not_called()

    def test_empty_search_skips_models(self):
        self.search.return_value=[]
        self.assertEqual(self.run_request()['steps'],['retrieve','escalate'])
        self.grader.assert_not_called()
        self.generator.assert_not_called()

    def test_human_request_skips_search_and_models(self):
        outcome=self.run_request('Please contact the Legal team')
        self.assertEqual(outcome['steps'],['escalate'])
        self.search.assert_not_called()
        self.grader.assert_not_called()
        self.generator.assert_not_called()

    def test_bad_citations_are_not_displayed(self):
        self.generator.return_value=Draft(action='answer',answer='Unsupported [FAKE]',citation_ids=['FAKE'])
        outcome=self.run_request()
        self.assertEqual(outcome['steps'],['retrieve','grade','generate','validate','escalate'])
        self.assertNotIn('Unsupported',outcome['answer'])

    def test_clarification_is_terminal_for_now(self):
        self.generator.return_value=Draft(action='clarify',answer='Which document?',citation_ids=[])
        outcome=self.run_request()
        self.assertEqual(outcome['action'],'clarify')
        self.assertEqual(outcome['steps'][-1],'check_claims')

    def test_model_can_escalate(self):
        self.generator.return_value=Draft(action='escalate',answer='Needs review',citation_ids=[])
        outcome=self.run_request()
        self.assertEqual(outcome['action'],'escalate')
        self.assertEqual(outcome['escalation_reason'],'Answer model requested Legal review')

    def test_node_failures_escalate_without_exposing_exception(self):
        for node in (self.search,self.grader,self.generator,self.validator):
            with self.subTest(node=node):
                node.side_effect=RuntimeError('SECRET-RAW-ERROR')
                outcome=self.run_request()
                self.assertEqual(outcome['action'],'escalate')
                self.assertTrue(outcome['error_message'])
                self.assertNotIn('SECRET-RAW-ERROR',str(outcome))
                node.side_effect=None

    def test_invocations_do_not_share_state(self):
        first=self.run_request()
        self.search.return_value=[]
        second=self.run_request('Unrelated question')
        self.assertEqual(first['action'],'answer')
        self.assertIsNone(second['grade'])
        self.assertIsNone(second['draft'])
        self.assertEqual(second['passages'],[])
        self.assertEqual(second['steps'],['retrieve','escalate'])

class EvidenceGateTests(unittest.TestCase):
    def test_human_decisions_and_inapplicable_policy_override_score(self):
        self.assertFalse(grade_passes(grade(evidence_support_percent=100, human_judgment_required=True), PASSAGES))
        self.assertFalse(grade_passes(grade(evidence_support_percent=100, policy_applicability="not_applicable"), PASSAGES))

    def test_graph_escalates_human_judgment_and_records_no_citation_check(self):
        from legal_assistant.workflow import run_workflow
        with patch('legal_assistant.workflow.get_retriever') as retriever,patch('legal_assistant.bot.grade_evidence',return_value=grade(human_judgment_required=True,human_judgment_reason='Legal approval required')),patch('legal_assistant.bot.generate_answer') as generator:
            retriever.return_value.search.return_value=PASSAGES
            result=run_workflow('Approve this agreement','ExampleCo (fictional)','2026.1')
            self.assertEqual(result['action'],'escalate')
            self.assertEqual(result['decision_reason'],'Legal approval required')
            self.assertEqual(result['citation_validation'],'Not performed')
            generator.assert_not_called()

class PerformanceTests(unittest.TestCase):
    def test_deduplication_preserves_qualifiers_and_policy_scope(self):
        from legal_assistant.retriever import deduplicate_passages
        base = {'id':'one', 'text':'Send the agreement to Legal.', 'metadata':{'source_type':'policy','organization':'ExampleCo','jurisdiction':'Global'}}
        duplicate = {**base, 'id':'two', 'text':'Question: How?\nAnswer: Send the agreement to Legal.'}
        variants = [
            {**base, 'id':'negation', 'text':'Do not send the agreement to Legal.'},
            {**base, 'id':'jurisdiction', 'metadata':{**base['metadata'],'jurisdiction':'US'}},
            {**base, 'id':'date', 'metadata':{**base['metadata'],'effective_from':'2027-01-01'}},
            {**base, 'id':'history', 'metadata':{**base['metadata'],'source_type':'reviewed_matter'}},
        ]
        actual = deduplicate_passages([base, duplicate, *variants], 6)
        self.assertEqual([p['id'] for p in actual], ['one','negation','jurisdiction','date','history'])
        self.assertEqual(len(deduplicate_passages([base, duplicate, *variants], 2)), 2)

    def test_timeout_settings_rotate_client_and_validate_retries(self):
        from legal_assistant import bot
        bot.clear_client_cache()
        self.addCleanup(bot.clear_client_cache)
        with patch.dict('os.environ', {'NEBIUS_API_KEY':'test-key','LLM_PROVIDER':'nebius','LLM_TIMEOUT_SECONDS':'30','LLM_MAX_RETRIES':'0'}), patch('openai.OpenAI') as factory:
            bot.create_client()
            self.assertEqual(factory.call_args.kwargs['timeout'],30)
            self.assertEqual(factory.call_args.kwargs['max_retries'],0)
            with patch.dict('os.environ',{'LLM_TIMEOUT_SECONDS':'15'}):
                bot.create_client()
            self.assertEqual(factory.call_count,2)
            with patch.dict('os.environ',{'LLM_MAX_RETRIES':'3'}):
                with self.assertRaises(ValueError):
                    bot.create_client()

    def test_client_is_reused_and_rotated_on_endpoint_change(self):
        from legal_assistant import bot
        bot.clear_client_cache()
        self.addCleanup(bot.clear_client_cache)
        with patch.dict('os.environ', {'LLM_PROVIDER':'nebius','NEBIUS_API_KEY':'test-key','LLM_BASE_URL':'https://api.tokenfactory.nebius.com/v1'}), patch('openai.OpenAI') as factory:
            first=bot.create_client()
            self.assertIs(bot.create_client(),first)
            self.assertEqual(factory.call_count,1)
            with patch.dict('os.environ',{'LLM_BASE_URL':'https://other.example.com/v1'}):
                bot.create_client()
            self.assertEqual(factory.call_count,2)
            first.close.assert_called_once()

    def test_manifest_switch_invalidates_cached_retriever(self):
        from legal_assistant import retriever
        retriever._cached_retriever.cache_clear()
        self.addCleanup(retriever._cached_retriever.cache_clear)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'storage').mkdir()
            manifest=root/'storage/index.json'
            manifest.write_text(json.dumps({'namespace':'one','documents':[]}))
            with patch('legal_assistant.retriever.ROOT',root),patch('legal_assistant.retriever.HybridRetriever',side_effect=[object(),object()]) as factory:
                first=retriever.get_retriever()
                self.assertIs(first,retriever.get_retriever())
                manifest.write_text(json.dumps({'namespace':'two','documents':[]}))
                second=retriever.get_retriever()
                self.assertIsNot(first,second)
                self.assertEqual(factory.call_count,2)

    def test_bm25_built_once_and_scope_remains_filtered(self):
        from legal_assistant import retriever
        from unittest.mock import MagicMock
        from legal_assistant.embeddings import MODEL_ID
        docs=load_documents(ROOT/'data')
        manifest={'vector_store':'pinecone','embedding_model':MODEL_ID,'index_name':'fixture','host':'fixture.pinecone.io','namespace':'test','documents':docs}
        index=MagicMock()
        index.query.return_value={'matches':[{'id':docs[0]['id'],'score':0.8}]}
        with patch('legal_assistant.retriever.BM25Okapi',wraps=retriever.BM25Okapi) as factory, patch('legal_assistant.retriever.connection_settings',return_value=('fixture','fixture.pinecone.io')):
            resource=retriever.HybridRetriever(manifest=manifest,index=index)
            built=factory.call_count
            results=resource.search('How do I request contract review?','ExampleCo (fictional)','2026.1')
            resource.search('Can I send an NDA?','ExampleCo (fictional)','2026.1')
            self.assertTrue(results)
            self.assertEqual(factory.call_count,built)
            self.assertEqual(resource.search('contract','Other organization','2026.1'),[])
            self.assertTrue(all(p['metadata']['organization']=='ExampleCo (fictional)' for p in results))

    def test_token_limits_are_applied_to_structured_calls(self):
        from legal_assistant import bot
        from unittest.mock import MagicMock
        client=MagicMock()
        client.chat.completions.parse.return_value=result(grade())
        with patch.dict('os.environ',{'GRADE_MAX_TOKENS':'700'}):
            bot.structured_call(client,'model','instructions',{},bot.EvidenceGrade)
        self.assertEqual(client.chat.completions.parse.call_args.kwargs['max_tokens'],700)

    def test_graph_records_elapsed_time_even_on_failure(self):
        from legal_assistant.workflow import run_workflow
        with patch('legal_assistant.workflow.get_retriever',side_effect=RuntimeError('offline')):
            outcome=run_workflow('question','ExampleCo (fictional)','2026.1')
        self.assertEqual(outcome['action'],'escalate')
        self.assertIn('retrieval',outcome['timings'])
        self.assertIn('total',outcome['timings'])
        self.assertNotIn('grading',outcome['timings'])
        self.assertGreaterEqual(outcome['timings']['total'],outcome['timings']['retrieval'])

class ConnectivityMessageTests(unittest.TestCase):
    def test_timeout_and_connection_errors_have_distinct_guidance(self):
        from legal_assistant.bot import pipeline_error_message
        Timeout = type('APITimeoutError', (Exception,), {})
        Connection = type('APIConnectionError', (Exception,), {})
        self.assertIn('request timeout', pipeline_error_message(Timeout()))
        self.assertIn('Connection to', pipeline_error_message(Connection()))
        self.assertNotIn('TLS', pipeline_error_message(Timeout()))

if __name__=='__main__': unittest.main()
