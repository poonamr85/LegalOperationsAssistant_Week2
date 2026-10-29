"""Claim validation, bounded SDK configuration and release routing regressions."""
import json
import unittest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError
from legal_assistant import bot
from legal_assistant.workflow import run_workflow
from tests.test_bot import PASSAGES, grade, result, supported_validation

DRAFT = bot.Draft(action='answer', answer='Use legal intake [LEGAL-FAQ-002:0].', citation_ids=['LEGAL-FAQ-002:0'])


class ClaimValidationTests(unittest.TestCase):
    def test_supported_claim_released_and_bad_verdicts_blocked(self):
        assessment = supported_validation()
        self.assertTrue(bot.claim_validation_passes(assessment, DRAFT, PASSAGES))
        for verdict in ['unsupported', 'contradicted']:
            assessment.claims[0].verdict = verdict
            self.assertFalse(bot.claim_validation_passes(assessment, DRAFT, PASSAGES))

    def test_incomplete_unknown_and_wrong_citation_evidence_rejected(self):
        for ids in [[], ['FAKE'], ['other:0']]:
            assessment = supported_validation()
            assessment.claims[0].evidence_ids = ids
            self.assertFalse(bot.claim_validation_passes(assessment, DRAFT,
                PASSAGES + [{'id':'other:0','text':'Unrelated'}]))
        assessment = supported_validation()
        assessment.all_claims_assessed = False
        self.assertFalse(bot.claim_validation_passes(assessment, DRAFT, PASSAGES))
        assessment = supported_validation()
        assessment.requires_human_decision = True
        self.assertFalse(bot.claim_validation_passes(assessment, DRAFT, PASSAGES))

    def test_omitted_line_and_invalid_unit_rejected(self):
        draft = DRAFT.model_copy(update={'answer': DRAFT.answer+'\nApproval takes two days.'})
        self.assertFalse(bot.claim_validation_passes(supported_validation(), draft, PASSAGES))
        assessment = supported_validation()
        assessment.claims[0].unit_id = 9
        self.assertFalse(bot.claim_validation_passes(assessment, DRAFT, PASSAGES))
        assessment = supported_validation()
        assessment.nonfactual_unit_ids = [0]
        self.assertFalse(bot.claim_validation_passes(assessment, DRAFT, PASSAGES))

    def test_no_claim_answer_fails_but_pure_clarification_can_pass(self):
        assessment = bot.ClaimValidation(all_claims_assessed=True, requires_human_decision=False,
                                        claims=[], nonfactual_unit_ids=[0])
        self.assertFalse(bot.claim_validation_passes(assessment, DRAFT, PASSAGES))
        clarification = bot.Draft(action='clarify',answer='Which agreement are you asking about?',citation_ids=[])
        self.assertTrue(bot.claim_validation_passes(assessment, clarification, PASSAGES))

    def test_schema_rejects_invalid_and_oversized_outputs(self):
        data = supported_validation().model_dump()
        for changes in [{'all_claims_assessed':'yes'}, {'claims':data['claims']*17}]:
            with self.assertRaises(ValidationError):
                bot.ClaimValidation.model_validate({**data, **changes})

    def test_one_bounded_call_uses_only_actual_selected_evidence(self):
        client = MagicMock()
        client.with_options.return_value.chat.completions.parse.return_value = result(supported_validation())
        with patch('legal_assistant.bot.create_client',return_value=client), patch.dict('os.environ',{
                'CLAIM_TIMEOUT_SECONDS':'8','CLAIM_MAX_TOKENS':'1000','CLAIM_MODEL':'validator-test'}):
            bot.validate_claim_support('Question', DRAFT, PASSAGES)
        client.with_options.assert_called_once_with(timeout=8.0,max_retries=0)
        parse = client.with_options.return_value.chat.completions.parse
        parse.assert_called_once()
        args = parse.call_args.kwargs
        self.assertEqual(args['model'],'validator-test')
        self.assertEqual(args['max_tokens'],1000)
        payload = json.loads(args['messages'][1]['content'])
        self.assertEqual(payload['evidence'],PASSAGES)
        self.assertEqual(payload['units'][0]['text'],DRAFT.answer)
        self.assertNotIn('grade',payload)

    def test_validator_timeout_or_refusal_has_no_retry_loop(self):
        for failure in [TimeoutError('PRIVATE'), ValueError('PRIVATE')]:
            client = MagicMock()
            parse = client.with_options.return_value.chat.completions.parse
            parse.side_effect = failure
            with patch('legal_assistant.bot.create_client',return_value=client):
                with self.assertRaises(type(failure)):
                    bot.validate_claim_support('Question', DRAFT, PASSAGES)
            parse.assert_called_once()

    def test_graph_hides_rejected_draft_and_records_added_time(self):
        for verdict in ['supported','unsupported','contradicted']:
            assessment = supported_validation()
            assessment.claims[0].verdict = verdict
            with patch('legal_assistant.workflow.get_retriever') as search, patch('legal_assistant.bot.grade_evidence',return_value=grade()), \
                    patch('legal_assistant.bot.generate_answer',return_value=DRAFT) as generator, \
                    patch('legal_assistant.bot.validate_claim_support',return_value=assessment) as validator:
                search.return_value.search.return_value = PASSAGES
                outcome = run_workflow('Question','ExampleCo (fictional)','2026.1')
            generator.assert_called_once()
            validator.assert_called_once_with('Question',DRAFT,PASSAGES)
            self.assertIn('claim validation',outcome['timings'])
            self.assertEqual(outcome['faithfulness_percent'],100.0 if verdict=='supported' else 0.0)
            if verdict == 'supported':
                self.assertEqual(outcome['answer'],DRAFT.answer)
            else:
                self.assertEqual(outcome['action'],'escalate')
                self.assertEqual(outcome['answer'],bot.LEGAL_CONTACT_MESSAGE)
                self.assertNotEqual(outcome['answer'],DRAFT.answer)

    def test_direct_escalations_and_invalid_citations_skip_validator(self):
        with patch('legal_assistant.workflow.get_retriever') as search, patch('legal_assistant.bot.grade_evidence',return_value=grade()) as grader, \
                patch('legal_assistant.bot.generate_answer',return_value=DRAFT) as generator, \
                patch('legal_assistant.bot.validate_claim_support') as validator:
            search.return_value.search.return_value = PASSAGES
            run_workflow('Please contact the Legal team','','')
            search.assert_not_called()
            generator.assert_not_called()
            grader.assert_not_called()
            validator.assert_not_called()
            for draft in [DRAFT.model_copy(update={'answer':'Wrong [FAKE]'}),
                          bot.Draft(action='escalate',answer='Contact Legal',citation_ids=[])]:
                generator.return_value = draft
                self.assertEqual(run_workflow('Question','ExampleCo (fictional)','2026.1')['action'],'escalate')
            validator.assert_not_called()

    def test_failed_validation_stays_distinct_from_low_evidence(self):
        with patch('legal_assistant.workflow.get_retriever') as search, patch('legal_assistant.bot.grade_evidence',return_value=grade()), \
                patch('legal_assistant.bot.generate_answer',return_value=DRAFT), \
                patch('legal_assistant.bot.validate_claim_support',side_effect=TimeoutError('PRIVATE')) as validator:
            search.return_value.search.return_value = PASSAGES
            outcome=run_workflow('Question','ExampleCo (fictional)','2026.1')
        self.assertEqual(outcome['action'],'escalate')
        self.assertEqual(outcome['failed_stage'],'claim validation')
        self.assertIsNone(outcome['faithfulness_percent'])
        self.assertNotIn('PRIVATE',str(outcome))
        validator.assert_called_once()


if __name__ == '__main__':
    unittest.main()
