"""Guardrails enforced in code, so they hold whatever the model decides to do.

These checks catch tests that pass without proving anything and keep the agent inside its own
folders. They are a quality gate, not a security sandbox: generated code still runs, so run the
agent on a disposable checkout or CI runner and review every PR.
"""

import ast
import re
from pathlib import Path

ALLOWED_IMPORTS = {"allure", "pytest", "re", "playwright.sync_api"}
BANNED_CALLS = {"eval", "exec", "compile", "open", "__import__", "breakpoint"}
# Fixed sleeps are the classic source of flaky tests; Playwright's expect() already auto-waits.
BANNED_METHODS = {"wait_for_timeout", "sleep"}
BANNED_PYTEST_CALLS = {"skip", "xfail", "fail", "importorskip"}
BANNED_MARKS = {"skip", "skipif", "xfail"}
MATCH_ANYTHING_PATTERNS = {"", ".*", ".+", "^.*$"}

PAGE_FILE = re.compile(r"^src/pages/[a-z0-9_]+\.py$")
TEST_FILE = re.compile(r"^tests/generated/[a-z0-9_]+_test\.py$")


class GuardrailError(Exception):
    """A tool call was blocked by a guardrail. The message is returned to the model."""


def resolve_write_path(root: Path, relative_path: str, created: set[str]) -> Path:
    """Allow new page objects and generated tests, and edits to files made in this run."""
    if not (PAGE_FILE.match(relative_path) or TEST_FILE.match(relative_path)):
        raise GuardrailError(
            f"Cannot write {relative_path}. Allowed: src/pages/<name>.py and "
            "tests/generated/<name>_test.py, using lowercase letters, digits and underscores."
        )
    target = _inside_root(root, relative_path)
    if target.exists() and relative_path not in created:
        raise GuardrailError(
            f"{relative_path} already exists and was not created in this run. "
            "Existing files are read-only; create a new file instead."
        )
    return target


def resolve_read_path(root: Path, relative_path: str) -> Path:
    """Allow reading project Python code and pyproject.toml, and nothing else."""
    in_code_folder = relative_path.startswith(("src/", "tests/")) and relative_path.endswith(".py")
    if not in_code_folder and relative_path != "pyproject.toml":
        raise GuardrailError(
            f"Cannot read {relative_path}. Allowed: .py files in src/ and tests/, pyproject.toml."
        )
    return _inside_root(root, relative_path)


def _inside_root(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root):
        raise GuardrailError(f"{relative_path} is outside the project.")
    return target


def validate_python(source: str, *, is_test: bool) -> list[str]:
    """Return every guardrail violation in the source. An empty list means it passed."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [f"Syntax error on line {error.lineno}: {error.msg}"]

    problems = []
    for node in ast.walk(tree):
        problems.extend(_check_node(node, is_test=is_test))
    if is_test:
        problems.extend(_check_test_functions(tree))
    return problems


def _check_node(node: ast.AST, *, is_test: bool) -> list[str]:
    problems = []
    if isinstance(node, ast.Import):
        problems.extend(_check_import(alias.name, node.lineno) for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        module = node.module or ""
        name = f".{module}" if node.level else module
        problems.append(_check_import(name, node.lineno))
    elif isinstance(node, ast.Call):
        problems.extend(_check_call(node))
    elif isinstance(node, ast.Attribute):
        if node.attr.startswith("__"):
            problems.append(f"Line {node.lineno}: dunder attribute {node.attr} is not allowed.")
        if (
            node.attr in BANNED_MARKS
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "mark"
        ):
            problems.append(f"Line {node.lineno}: pytest.mark.{node.attr} hides failures.")
    elif is_test and isinstance(node, ast.Try):
        problems.append(f"Line {node.lineno}: try/except can swallow assertion failures.")
    elif isinstance(node, ast.Assert) and isinstance(node.test, ast.Constant):
        problems.append(f"Line {node.lineno}: asserting a constant proves nothing.")
    return [problem for problem in problems if problem]


def _check_import(name: str, line: int) -> str:
    if name in ALLOWED_IMPORTS or name.startswith("src."):
        return ""
    return f"Line {line}: import of {name} is not allowed."


def _check_call(node: ast.Call) -> list[str]:
    func = node.func
    if isinstance(func, ast.Name) and func.id in BANNED_CALLS:
        return [f"Line {node.lineno}: {func.id}() is not allowed."]
    if not isinstance(func, ast.Attribute):
        return []
    if func.attr in BANNED_METHODS:
        return [f"Line {node.lineno}: {func.attr}() is a fixed wait; use expect() instead."]
    if isinstance(func.value, ast.Name):
        if func.value.id == "pytest" and func.attr in BANNED_PYTEST_CALLS:
            return [f"Line {node.lineno}: pytest.{func.attr}() is not allowed."]
        if func.value.id == "re" and func.attr == "compile" and _matches_anything(node):
            return [f"Line {node.lineno}: a pattern that matches anything proves nothing."]
    return []


def _matches_anything(node: ast.Call) -> bool:
    if not node.args or not isinstance(node.args[0], ast.Constant):
        return False
    return node.args[0].value in MATCH_ANYTHING_PATTERNS


def _check_test_functions(tree: ast.AST) -> list[str]:
    tests = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    if not tests:
        return ["The file has no test_ functions."]
    return [
        f"Line {test.lineno}: {test.name} has no assertion; add expect(...) or assert."
        for test in tests
        if not _has_assertion(test)
    ]


def _has_assertion(test: ast.FunctionDef) -> bool:
    for node in ast.walk(test):
        if isinstance(node, ast.Assert) and not isinstance(node.test, ast.Constant):
            return True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "expect"
        ):
            return True
    return False
