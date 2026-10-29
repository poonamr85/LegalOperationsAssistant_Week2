"""Evaluation metrics and reporting regressions; no provider calls."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from legal_assistant import bot
from evaluation.eval_set import CASES
from evaluation.resolution_eval import (metrics, make_row, write_reports, assess_answer, QualityAssessment, CriterionResult)
from tests.test_bot import PASSAGES, supported_validation, result


def answered_state():
    draft=bot.Draft(action='answer',answer='Use intake [LEGAL-FAQ-002:0].',citation_ids=['LEGAL-FAQ-002:0'])
    return {'action':'answer','answer':draft.answer,'draft':draft,'claim_validation':supported_validation(),
            'passages':PASSAGES,'selected_passages':PASSAGES,'faithfulness_percent':100,
            'steps':['retrieve','grade','generate','validate','check_claims']}


def complete_assessment(case):
    return QualityAssessment(criteria=[CriterionResult(criterion_id=i,met=True,
        evidence_ids=['LEGAL-FAQ-002:0'],reason='Supported') for i in range(len(case['required_guidance']))],
        contradicts_reference=False,unsupported_claim_present=False)


class ResolutionEvaluationTests(unittest.TestCase):
    def test_dataset_has_twenty_unique_questions_and_explicit_rubrics(self):
        self.assertEqual(len(CASES),20)
        self.assertEqual(len({c['question'] for c in CASES}),20)
        self.assertEqual(len({c['id'] for c in CASES}),20)
        self.assertEqual(sum(c['expected_action']=='answer' for c in CASES),16)
        self.assertTrue(all(c['required_guidance'] for c in CASES if c['expected_action']=='answer'))

    def test_answer_is_not_resolution_without_complete_quality_and_live_validation(self):
        case=next(c for c in CASES if c['expected_action']=='answer')
        for quality in ['partial','incorrect','unassessed']:
            self.assertFalse(make_row(case,answered_state(),1,quality,None)['resolved'])
        self.assertTrue(make_row(case,answered_state(),1,'complete',complete_assessment(case))['resolved'])
        state=answered_state();state['claim_validation']=None
        self.assertFalse(make_row(case,state,1,'complete',None)['resolved'])
        self.assertFalse(make_row(case,answered_state(),1,'complete',None,'evaluator failed')['resolved'])

    def test_processing_failure_not_correct_or_unnecessary_escalation(self):
        answer_case=next(c for c in CASES if c['expected_action']=='answer')
        escalation_case=next(c for c in CASES if c['expected_action']=='escalate')
        failed={'action':'escalate','error_message':'Provider unavailable','failed_stage':'evidence grading'}
        a=make_row(answer_case,failed,2,'not_answered',None)
        b=make_row(escalation_case,failed,2,'not_applicable',None)
        self.assertFalse(b['correct_escalation'])
        m=metrics([a,b])
        self.assertEqual(m['processing_failure_rate_percent'],100)
        self.assertEqual(m['unnecessary_escalation_rate_percent'],0)
        self.assertEqual(m['correct_escalation_rate_percent'],0)
        self.assertEqual(m['automated_verified_resolution_rate_percent'],0)

    def test_p95_denominators_and_separate_offline_time(self):
        case=next(c for c in CASES if c['expected_action']=='answer')
        rows=[make_row(case,answered_state(),n,'complete',complete_assessment(case),quality_seconds=99) for n in range(1,21)]
        m=metrics(rows)
        self.assertEqual(m['median_response_seconds'],10.5)
        self.assertEqual(m['p95_response_seconds'],19)
        self.assertEqual(m['automated_verified_resolution_rate_percent'],100)
        self.assertIsNone(m['correct_escalation_rate_percent'])
        self.assertIsNone(metrics([])['median_response_seconds'])

    def test_reports_record_expected_actual_quality_reason_and_latency(self):
        row=make_row(CASES[0],answered_state(),3,'complete',None)
        report={'results':[row],'metrics':metrics([row]),'status':'complete','planned_queries':1,
                'created_at':'2026-10-23','limitations':'Automated development test'}
        with tempfile.TemporaryDirectory() as folder:
            write_reports(report,Path(folder))
            saved=json.loads((Path(folder)/'resolution_report.json').read_text())
            self.assertEqual(saved['results'][0]['expected_action'],'answer')
            self.assertEqual(saved['results'][0]['actual_action'],'answer')
            self.assertEqual(saved['results'][0]['response_latency_seconds'],3)
            self.assertTrue((Path(folder)/'resolution_report.csv').exists())
            self.assertIn('Automated development evaluation',(Path(folder)/'resolution_report.html').read_text())

    def test_quality_evaluator_rejects_missing_criteria_and_unknown_references(self):
        case=CASES[0]
        client=MagicMock()
        parse=client.with_options.return_value.chat.completions.parse
        assessment=QualityAssessment(criteria=[],contradicts_reference=False,unsupported_claim_present=False)
        parse.return_value=result(assessment)
        with patch('legal_assistant.bot.create_client',return_value=client):
            with self.assertRaises(ValueError):assess_answer(case,answered_state())
            assessment.criteria=[CriterionResult(criterion_id=i,met=True,evidence_ids=['FAKE'],reason='bad') for i in range(len(case['required_guidance']))]
            with self.assertRaises(ValueError):assess_answer(case,answered_state())


if __name__=='__main__':unittest.main()
