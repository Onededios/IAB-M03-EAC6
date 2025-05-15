@echo off
echo Running code linter on folder
python -m pylint . > lint_result.txt
echo Lint file results saved at lint_result.txt