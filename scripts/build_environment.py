"""Build an isolated deployment environment without modifying system Python."""
import os
from pathlib import Path
import subprocess
import venv

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    python = root / '.venv/bin/python'
    if not python.exists():
        # Bootstrap pip explicitly below; the base toolchain need not include pip.
        venv.EnvBuilder(with_pip=False).create(root / '.venv')
    env = {**os.environ, 'PIP_USER': 'false'}
    commands = [
        [str(python), '-m', 'ensurepip', '--upgrade'],
        [str(python), '-m', 'pip', 'install', '-r', 'requirements.txt'],
        [str(python), '-m', 'pip', 'check'],
    ]
    for command in commands:
        subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == '__main__':
    build()
