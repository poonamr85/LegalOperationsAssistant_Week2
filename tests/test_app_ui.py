"""Headless UI regression checks; no provider calls or server required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from legal_assistant.knowledge_base import ROOT


class AssistantUITests(unittest.TestCase):
    def open_assistant(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=15).run()
        app.switch_page('legal_assistant/app_pages/assistant.py').run()
        self.assertFalse(app.exception)
        return app

    def test_answer_back_and_clear_preserve_navigation_behavior(self):
        app = self.open_assistant()
        self.assertFalse(any(button.label == 'Back' for button in app.button))
        with patch('legal_assistant.workflow.run_workflow', return_value={
            'action': 'answer', 'answer': 'Use intake [LEGAL-FAQ-002:0].'
        }) as workflow:
            app.chat_input[0].set_value('How do I request contract review?').run()
        self.assertFalse(app.exception)
        workflow.assert_called_once()
        app.button(key='back_to_overview').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Legal guidance.' in title.value for title in app.title))
        self.assertEqual(len(app.session_state['messages']), 2)
        app.switch_page('legal_assistant/app_pages/assistant.py').run()
        next(button for button in app.button if button.label == 'Clear conversation').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['messages'], [])
        self.assertFalse(any(button.label == 'Back' for button in app.button))

    def test_no_index_and_human_request_are_routed_by_graph(self):
        # No scope: setup guidance. A human request still bypasses search/models.
        with tempfile.TemporaryDirectory() as tmp, patch('legal_assistant.knowledge_base.ROOT', Path(tmp)), \
                patch('legal_assistant.workflow.get_retriever') as retriever, \
                patch('legal_assistant.bot.grade_evidence') as grader, patch('legal_assistant.bot.generate_answer') as generator:
            app = self.open_assistant()
            app.chat_input[0].set_value('How do I request contract review?').run()
            self.assertFalse(app.exception)
            self.assertIn('administrator', app.session_state['messages'][-1]['content'])
            app.chat_input[0].set_value('Please contact the Legal team').run()
            self.assertFalse(app.exception)
            self.assertIn('contact the Legal team', app.session_state['messages'][-1]['content'])
            self.assertIn('Employee requested', app.session_state['handoff']['reason'])
            retriever.assert_not_called()
            grader.assert_not_called()
            generator.assert_not_called()

    def test_unexpected_workflow_failure_is_sanitized(self):
        app = self.open_assistant()
        with patch('legal_assistant.workflow.run_workflow', side_effect=RuntimeError('PRIVATE-DETAIL')):
            app.chat_input[0].set_value('How do I request review?').run()
        self.assertFalse(app.exception)
        self.assertTrue(app.warning)
        self.assertIn('contact the Legal team', app.session_state['messages'][-1]['content'])
        self.assertNotIn('PRIVATE-DETAIL', str(app.session_state['messages']))
        self.assertNotIn('PRIVATE-DETAIL', str(app.warning))


if __name__ == '__main__':
    unittest.main()
