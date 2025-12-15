from dataclasses import dataclass
from typing import List


@dataclass
class Task:
    name: str
    prompt: str
    canonical_solution: str
    test_code: str


# 极简任务集合：字符级代码生成容易探索
TASKS: List[Task] = [
    Task(
        name="abs_value",
        prompt="实现一个函数 abs_value(x)，返回数字 x 的绝对值。",
        canonical_solution="""
def abs_value(x):
    return x if x >= 0 else -x
""",
        test_code="""
import pytest
from solution import abs_value


@pytest.mark.parametrize("x,expected", [(-5, 5), (0, 0), (3.5, 3.5)])
def test_abs_value(x, expected):
    assert abs_value(x) == expected
""",
    ),
    Task(
        name="reverse_string",
        prompt="实现 reverse_string(s)，返回字符串 s 的反转。",
        canonical_solution="""
def reverse_string(s: str) -> str:
    return s[::-1]
""",
        test_code="""
import pytest
from solution import reverse_string


@pytest.mark.parametrize(
    "s,expected",
    [("", ""), ("a", "a"), ("abc", "cba"), ("racecar", "racecar")],
)
def test_reverse_string(s, expected):
    assert reverse_string(s) == expected
""",
    ),
    Task(
        name="factorial",
        prompt="实现 factorial(n)，返回非负整数 n 的阶乘。",
        canonical_solution="""
def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
""",
        test_code="""
import pytest
from solution import factorial


def test_small_values():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(5) == 120


def test_invalid():
    with pytest.raises(ValueError):
        factorial(-1)
""",
    ),
    Task(
        name="is_palindrome",
        prompt="实现 is_palindrome(s)，忽略空格与大小写判断回文。",
        canonical_solution="""
def is_palindrome(s: str) -> bool:
    cleaned = "".join(ch.lower() for ch in s if not ch.isspace())
    return cleaned == cleaned[::-1]
""",
        test_code="""
import pytest
from solution import is_palindrome


@pytest.mark.parametrize(
    "s,expected",
    [
        ("", True),
        ("a", True),
        ("Race car", True),
        ("hello", False),
        ("Never odd or even", True),
    ],
)
def test_is_palindrome(s, expected):
    assert is_palindrome(s) is expected
""",
    ),
    Task(
        name="count_vowels",
        prompt="实现 count_vowels(s)，返回字符串中元音字母的数量（aeiou，忽略大小写）。",
        canonical_solution="""
def count_vowels(s: str) -> int:
    vowels = set("aeiou")
    return sum(1 for ch in s.lower() if ch in vowels)
""",
        test_code="""
import pytest
from solution import count_vowels


@pytest.mark.parametrize(
    "s,expected",
    [
        ("", 0),
        ("bcd", 0),
        ("abc", 1),
        ("AEIOU", 5),
        ("Hello World", 3),
    ],
)
def test_count_vowels(s, expected):
    assert count_vowels(s) == expected
""",
    ),
]
