import string
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tasks.tasks import Task, TASKS


class CodeGenEnv(gym.Env):
    """
    一个极简的字符级代码生成环境：
    - 观测：dict(prompt, code)，prompt 提供任务描述，code 为当前已生成代码。
    - 动作：离散字符表；特殊 token <EOS> 代表结束。
    - 奖励：生成结束后运行对应任务的 pytest，全部通过记为 1，否则 0。
    """

    metadata = {"render_modes": []}

    def __init__(self, tasks=None, max_steps: int = 200):
        super().__init__()
        self.tasks = tasks or TASKS
        self.max_steps = max_steps

        # 限制字符表，降低搜索空间；最后一个 token 为 <EOS>。
        # 同时包含任务 prompt 中的字符（有中文），保证编码不报错。
        base_chars = string.ascii_letters + string.digits + " _():.,\n+-*/=<>'\"[]{}#"
        extra_chars = set()
        for t in self.tasks:
            extra_chars.update(t.prompt)
        self.vocab = list(dict.fromkeys(base_chars + "".join(sorted(extra_chars))))  # 去重保序
        self.eos_token = "<EOS>"
        self.vocab.append(self.eos_token)

        self.action_space = spaces.Discrete(len(self.vocab))
        charset = list(string.printable) + [self.eos_token]
        self.observation_space = spaces.Dict(
            {
                "prompt": spaces.Text(max_length=256, min_length=1, charset=charset),
                "code": spaces.Text(max_length=max_steps + 8, min_length=0, charset=charset),
            }
        )

        self._rng = np.random.default_rng()
        self._task: Optional[Task] = None
        self._code_buffer: str = ""
        self._step_count: int = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._task = self._rng.choice(self.tasks)
        self._code_buffer = ""
        self._step_count = 0

        obs = {"prompt": self._task.prompt, "code": self._code_buffer}
        info = {"task": self._task.name}
        return obs, info

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        token = self.vocab[action]
        terminated = False
        truncated = False
        reward = 0.0

        if token == self.eos_token:
            terminated = True
        else:
            self._code_buffer += token
            self._step_count += 1
            if self._step_count >= self.max_steps:
                truncated = True

        if terminated or truncated:
            reward, passed = self._evaluate_code(self._code_buffer, self._task)
            info = {"task": self._task.name, "passed": passed}
        else:
            info = {"task": self._task.name}

        obs = {"prompt": self._task.prompt, "code": self._code_buffer}
        return obs, reward, terminated, truncated, info

    def render(self):
        print("Prompt:", self._task.prompt if self._task else None)
        print("Code:\n", self._code_buffer)

    def _evaluate_code(self, code_str: str, task: Task) -> Tuple[float, bool]:
        """写入临时代码并运行 task.test_code，返回 (reward, passed)。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "solution.py").write_text(code_str, encoding="utf-8")
            (tmp_path / "test_solution.py").write_text(task.test_code, encoding="utf-8")

            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--disable-warnings", "--maxfail=1"],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                text=True,
            )
            passed = result.returncode == 0
            reward = 1.0 if passed else 0.0
            return reward, passed
