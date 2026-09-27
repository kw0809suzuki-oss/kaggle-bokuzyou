"""Build a single Python submission, with the pinned Teacher embedded.

Usage: python build_astra_flow_submission_v0.py /path/to/submission.py
Official kaggle-environments 1.32.7 remains a runtime dependency.
"""
from pathlib import Path
import sys


def build(destination):
    root = Path(__file__).resolve().parent
    source = (root / 'astra_flow_terminal_model_v0.py').read_text()
    teacher = (root / 'astra_flow_vendor/seyamalam_v21.py').read_text()
    needle = 'TEACHER_SOURCE = Path(__file__).with_name("astra_flow_vendor").joinpath("seyamalam_v21.py").read_text()'
    assert source.count(needle) == 1
    source = source.replace(needle, 'TEACHER_SOURCE = ' + repr(teacher))
    compile(source, str(destination), 'exec')
    Path(destination).write_text(source)


if __name__ == '__main__':
    build(sys.argv[1])
