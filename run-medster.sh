#!/bin/bash
# Medster-local-LLM launcher script
# Workaround for Python 3.13 .pth file import issues with uv

export PYTHONPATH="/Users/sbm4_mac/Desktop/Medster-local-LLM/src:$PYTHONPATH"
cd /Users/sbm4_mac/Desktop/Medster-local-LLM
exec uv run python -m medster.cli "$@"
