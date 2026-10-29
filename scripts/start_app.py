"""Prepare once per server startup, then replace this process with Streamlit."""
import os
from pathlib import Path
import sys


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    from scripts.prepare_app import main as prepare
    prepare()
    os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", "app.py",
                            "--server.address=0.0.0.0", "--server.port=8501",
                            "--server.headless=true"])


if __name__ == "__main__":
    main()
