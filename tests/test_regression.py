"""Bad cases frozen from experiments must keep passing on B4."""
import pytest

from evaluation_platform import regression
from evaluation_platform.runner import run_task

CASES = regression.load()


@pytest.mark.parametrize("task", CASES, ids=[c["task_id"] for c in CASES])
def test_regression_case(task):
    r = run_task(task, "B4", persist=False)
    assert r["passed"], f"{task['task_id']} ({task['regression_reason']}): {r['error_category']}"
