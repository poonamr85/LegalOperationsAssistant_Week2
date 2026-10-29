"""Offline grouped review and new fictional policy data checks."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from evaluation.grouped_eval_report import ROOT, build_report
from legal_assistant.knowledge_base import load_documents


class GroupedReviewTests(unittest.TestCase):
    def test_dataset_groups_and_unrun_examples(self):
        cases=json.loads((ROOT/'evaluation/scenario_eval_set.json').read_text())['cases']
        self.assertEqual(len(cases),25)
        self.assertEqual(len({c['id'] for c in cases}),25)
        self.assertEqual(len({c['question'] for c in cases}),25)
        self.assertEqual(sum(c['category']=='ambiguous' for c in cases),5)
        self.assertTrue(all(c['expected_response_example'] for c in cases))

    def test_new_policies_ingested_under_existing_scope(self):
        documents=load_documents(ROOT/'data')
        added=[d for d in documents if d['metadata']['source_id'].startswith('POL-CLAR-')]
        self.assertEqual(len(added),3)
        self.assertTrue(all(d['metadata']['organization']=='ExampleCo (fictional)' for d in added))
        self.assertTrue(all(d['metadata']['policy_version']=='2026.1' for d in added))
        self.assertEqual(len(documents),49)

    def test_saved_results_reused_without_network_and_no_invented_ambiguity_stats(self):
        with tempfile.TemporaryDirectory() as directory, patch('socket.create_connection',side_effect=AssertionError('Network must not be used')),contextlib.redirect_stdout(io.StringIO()):
            report=build_report(output=Path(directory))
            self.assertEqual(report['categories']['expected_answer']['handling_rate_percent'],100)
            self.assertEqual(report['categories']['legal_redirect']['saved_results_count'],4)
            self.assertEqual(report['categories']['ambiguous']['not_run_count'],5)
            self.assertIsNone(report['categories']['ambiguous']['handling_rate_percent'])
            for case in report['results']:
                if case['category']=='ambiguous':
                    self.assertIsNone(case['actual_answer'])
                    self.assertIsNone(case['response_latency_seconds'])
            self.assertIn('before new clarification policies',report['notice'])
            self.assertFalse(report['baseline_matches_current_index'])
            html=(Path(directory)/'grouped_report.html').read_text()
            self.assertIn('not run',html)
            self.assertTrue((Path(directory)/'grouped_report.csv').exists())


if __name__=='__main__':unittest.main()
