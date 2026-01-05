"""
Utilities to load MBPP tasks from Hugging Face datasets.
"""

from __future__ import annotations

import random
from typing import List

from tasks.tasks import Task


def build_test_code(test_setup_code: str, test_list: List[str]) -> str:
    setup = (test_setup_code or "").strip()
    lines = [
        "import pytest",
        "from solution import *",
    ]
    if setup:
        lines.append(setup)
    lines.append("")
    lines.append("def test_mbpp():")
    for t in test_list:
        lines.append(f"    {t.strip()}")
    lines.append("")
    return "\n".join(lines)


def load_mbpp_tasks(
    split: str = "train",
    max_samples: int = 0,
    seed: int = 42,
    use_challenge_tests: bool = False,
) -> List[Task]:
    from datasets import load_dataset

    ds = load_dataset("mbpp", split=split)
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    if max_samples > 0:
        indices = indices[:max_samples]

    tasks: List[Task] = []
    for i in indices:
        item = ds[int(i)]
        tests = item["challenge_test_list"] if use_challenge_tests else item["test_list"]
        test_code = build_test_code(item["test_setup_code"], tests)
        tasks.append(
            Task(
                name=f"mbpp_{item['task_id']}",
                prompt=item["text"].strip(),
                canonical_solution=item["code"].strip(),
                test_code=test_code,
            )
        )
    return tasks
