"""Launcher for the evaluation harness: python cli.py run | eval | compare."""
import sys

from rag_eval.cli import main

if __name__ == '__main__':
    sys.exit(main())
