import py_compile
py_compile.compile("kaggle_submission_return_aware_v0/submission.py", doraise=True)
from kaggle_submission_return_aware_v0 import submission
assert callable(submission.agent)
print("submission import: PASS")
