#!/bin/bash
# Every candidate terminates before a new, unmodified hidden-test process starts.
set -u
cd /home/user
case_name="$1"
seed="${2:-0}"
python -m redteam.control reset
cp "redteam/$case_name/candidate.py" method/audit_candidate.py
python -m method.audit_candidate --seed "$seed" > "redteam/$case_name/method.log" 2>&1
method_exit=$?
bash scripts/evaluate_train.sh > "redteam/$case_name/train.log" 2>&1
train_exit=$?
# No arguments, no environment changes, no evaluator modifications or imports.
bash scripts/evaluate_test.sh > "redteam/$case_name/test.log" 2>&1
test_exit=$?
python -m redteam.control save "$case_name" "$train_exit" "$test_exit" "$method_exit"
echo "$case_name method=$method_exit train=$train_exit test=$test_exit"
python -m redteam.control restore
