@echo off
echo Generating docs from python docstrings
if not exist "docs" mkdir "docs"
python -m pydoc functions > docs/functions
echo Docs saved at ./docs