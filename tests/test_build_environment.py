"""Deployment bootstrap checks; package installs are mocked."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from scripts.build_environment import build


class DeploymentBuildTests(unittest.TestCase):
    def test_creates_environment_and_preserves_package_firewall_settings(self):
        with tempfile.TemporaryDirectory() as temp, patch('venv.EnvBuilder') as builder, \
             patch('subprocess.run') as run, patch.dict('os.environ', {'PIP_USER': 'true', 'PIP_INDEX_URL': 'https://example.invalid/simple'}):
            build(Path(temp))
            builder.return_value.create.assert_called_once_with(Path(temp)/'.venv')
            self.assertEqual(run.call_count, 3)
            for call in run.call_args_list:
                self.assertEqual(call.kwargs['env']['PIP_USER'], 'false')
                self.assertEqual(call.kwargs['env']['PIP_INDEX_URL'], 'https://example.invalid/simple')
                self.assertTrue(call.kwargs['check'])
            self.assertIn('ensurepip',run.call_args_list[0].args[0])

    def test_existing_environment_is_not_recreated(self):
        with tempfile.TemporaryDirectory() as temp, patch('venv.EnvBuilder') as builder, patch('subprocess.run'):
            python=Path(temp)/'.venv/bin/python'
            python.parent.mkdir(parents=True)
            python.touch()
            build(Path(temp))
            builder.assert_not_called()

    def test_failed_install_stops_build(self):
        with tempfile.TemporaryDirectory() as temp, patch('venv.EnvBuilder'), \
             patch('subprocess.run',side_effect=[None,subprocess.CalledProcessError(1,'pip')]) as run:
            with self.assertRaises(subprocess.CalledProcessError):build(Path(temp))
            self.assertEqual(run.call_count,2)
