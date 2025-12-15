import subprocess
import sys
import tempfile
from pathlib import Path

from tasks.tasks import TASKS


def run_task_tests(task):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "solution.py").write_text(task.canonical_solution, encoding="utf-8")
        (tmp_path / "test_solution.py").write_text(task.test_code, encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--disable-warnings", "--maxfail=1"],
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"Task {task.name} canonical solution failed tests.\n{result.stdout}"
            )


def test_all_tasks_pass():
    for task in TASKS:
        run_task_tests(task)
