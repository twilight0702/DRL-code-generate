import random
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
    Task(
        name="max_of_three",
        prompt="实现 max_of_three(a, b, c)，返回三个数中的最大值。",
        canonical_solution="""
def max_of_three(a, b, c):
    return max(a, b, c)
""",
        test_code="""
import pytest
from solution import max_of_three


@pytest.mark.parametrize(
    "a,b,c,expected",
    [(1, 2, 3, 3), (5, 2, 4, 5), (-1, -2, -3, -1)],
)
def test_max_of_three(a, b, c, expected):
    assert max_of_three(a, b, c) == expected
""",
    ),
    Task(
        name="clamp",
        prompt="实现 clamp(x, lo, hi)，将 x 限制在 [lo, hi] 范围内。",
        canonical_solution="""
def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
""",
        test_code="""
import pytest
from solution import clamp


@pytest.mark.parametrize(
    "x,lo,hi,expected",
    [(-1, 0, 10, 0), (5, 0, 10, 5), (12, 0, 10, 10)],
)
def test_clamp(x, lo, hi, expected):
    assert clamp(x, lo, hi) == expected
""",
    ),
    Task(
        name="sum_list",
        prompt="实现 sum_list(nums)，返回数值列表的元素和。",
        canonical_solution="""
def sum_list(nums):
    return sum(nums)
""",
        test_code="""
import pytest
from solution import sum_list


@pytest.mark.parametrize(
    "nums,expected",
    [([], 0), ([1, 2, 3], 6), ([-1, 1, 4], 4)],
)
def test_sum_list(nums, expected):
    assert sum_list(nums) == expected
""",
    ),
    Task(
        name="unique_count",
        prompt="实现 unique_count(items)，返回列表中不重复元素的数量。",
        canonical_solution="""
def unique_count(items):
    return len(set(items))
""",
        test_code="""
import pytest
from solution import unique_count


@pytest.mark.parametrize(
    "items,expected",
    [([], 0), ([1, 1, 2], 2), (["a", "b", "a"], 2)],
)
def test_unique_count(items, expected):
    assert unique_count(items) == expected
""",
    ),
    Task(
        name="is_even",
        prompt="实现 is_even(n)，判断整数 n 是否为偶数。",
        canonical_solution="""
def is_even(n: int) -> bool:
    return n % 2 == 0
""",
        test_code="""
import pytest
from solution import is_even


@pytest.mark.parametrize(
    "n,expected",
    [(0, True), (1, False), (2, True), (-3, False)],
)
def test_is_even(n, expected):
    assert is_even(n) is expected
""",
    ),
    Task(
        name="second_largest",
        prompt="实现 second_largest(nums)，返回列表中按数值排序的第二大值（允许重复）。",
        canonical_solution="""
def second_largest(nums):
    sorted_nums = sorted(nums, reverse=True)
    return sorted_nums[1]
""",
        test_code="""
import pytest
from solution import second_largest


@pytest.mark.parametrize(
    "nums,expected",
    [([1, 2, 3], 2), ([5, 5, 1], 5), ([-1, -2, -3], -2)],
)
def test_second_largest(nums, expected):
    assert second_largest(nums) == expected
""",
    ),
    Task(
        name="count_words",
        prompt="实现 count_words(s)，返回字符串中的单词数量（按空白分割）。",
        canonical_solution="""
def count_words(s: str) -> int:
    return len(s.split())
""",
        test_code="""
import pytest
from solution import count_words


@pytest.mark.parametrize(
    "s,expected",
    [("", 0), ("hello", 1), ("hello world", 2), ("  a  b ", 2)],
)
def test_count_words(s, expected):
    assert count_words(s) == expected
""",
    ),
    Task(
        name="fizzbuzz_str",
        prompt="实现 fizzbuzz_str(n)，返回 Fizz/Buzz/FizzBuzz 或数字字符串。",
        canonical_solution="""
def fizzbuzz_str(n: int) -> str:
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)
""",
        test_code="""
import pytest
from solution import fizzbuzz_str


@pytest.mark.parametrize(
    "n,expected",
    [(1, "1"), (3, "Fizz"), (5, "Buzz"), (15, "FizzBuzz"), (16, "16")],
)
def test_fizzbuzz_str(n, expected):
    assert fizzbuzz_str(n) == expected
""",
    ),
]


def split_tasks(
    tasks: List[Task], train_ratio: float = 0.8, seed: int = 42
) -> tuple[List[Task], List[Task]]:
    """Deterministically split tasks into train/val subsets."""
    if not tasks:
        return [], []
    if len(tasks) == 1:
        return tasks[:], tasks[:]
    rng = random.Random(seed)
    indices = list(range(len(tasks)))
    rng.shuffle(indices)
    split_idx = int(len(tasks) * train_ratio)
    split_idx = max(1, min(len(tasks) - 1, split_idx))
    train = [tasks[i] for i in indices[:split_idx]]
    val = [tasks[i] for i in indices[split_idx:]]
    return train, val
