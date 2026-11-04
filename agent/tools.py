"""The tools the test-writing agent can call, and their implementations."""

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from agent.safety import (
    TEST_FILE,
    GuardrailError,
    resolve_read_path,
    resolve_write_path,
    validate_python,
)

DEFAULT_BASE_URL = "https://www.saucedemo.com/"
ACTION_TIMEOUT_MS = 5_000
TEST_RUN_TIMEOUT_SECONDS = 240
MAX_TOOL_OUTPUT_CHARS = 20_000
# Generated tests run as subprocesses and have no reason to see these.
SECRET_ENV_PREFIXES = ("ANTHROPIC_", "GITHUB_", "GH_")
COLLECT_TEST_IDS_JS = """
elements => elements.map(element => ({
  id: element.getAttribute('data-test'),
  tag: element.tagName.toLowerCase(),
  text: (element.innerText || element.value || '').trim().slice(0, 80),
}))
"""

TOOL_DEFINITIONS = [
    {
        "name": "list_files",
        "description": "List the Python files in src/ and tests/.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": "Read a .py file from src/ or tests/, or pyproject.toml.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path from the repo root."}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "inspect_page",
        "description": (
            "Open a page of the application under test in a fresh browser, optionally perform "
            "actions, and return its URL, data-test elements and accessibility snapshot. Each "
            "call starts a new session, so repeat earlier actions to reach a later state."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to the base URL, such as inventory.html.",
                },
                "username": {
                    "type": "string",
                    "description": "User to start logged in as, or an empty string for none.",
                },
                "actions": {
                    "type": "array",
                    "description": "Actions to perform in order after the page loads.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": ["click", "fill"]},
                            "test_id": {"type": "string"},
                            "value": {
                                "type": "string",
                                "description": "Text for fill; empty string for click.",
                            },
                        },
                        "required": ["op", "test_id", "value"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["path", "username", "actions"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create a page object at src/pages/<name>.py or a test at "
            "tests/generated/<name>_test.py. Existing project files cannot be changed. "
            "Weak tests, such as ones without assertions, are rejected with the reasons."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_tests",
        "description": "Run the test files created in this session and return pytest's output.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "finish",
        "description": (
            "End the session. The tests are re-run independently afterwards, so this status "
            "is checked. Use passed when all new tests pass, bug_suspected when a test fails "
            "because the application misbehaves, and blocked when you could not write the test."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["passed", "bug_suspected", "blocked"]},
                "summary": {
                    "type": "string",
                    "description": "What was tested and, for a bug, expected versus actual.",
                },
            },
            "required": ["status", "summary"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class PytestRun:
    exit_code: int
    output: str


@dataclass
class Toolbox:
    root: Path
    base_url: str = DEFAULT_BASE_URL
    created: set[str] = field(default_factory=set)

    def dispatch(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        """Run a tool and return (result text, is_error). Failures go back to the model."""
        handlers = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "inspect_page": self.inspect_page,
            "write_file": self.write_file,
            "run_tests": self.run_tests_tool,
        }
        handler = handlers.get(name)
        if handler is None:
            return f"Unknown tool: {name}", True
        try:
            return handler(**tool_input), False
        except (GuardrailError, OSError, PlaywrightError, TypeError) as error:
            return str(error), True

    def list_files(self) -> str:
        files = [
            path.relative_to(self.root).as_posix()
            for folder in ("src", "tests")
            for path in sorted((self.root / folder).rglob("*.py"))
        ]
        return "\n".join(files)

    def read_file(self, path: str) -> str:
        return resolve_read_path(self.root, path).read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> str:
        target = resolve_write_path(self.root, path, self.created)
        problems = validate_python(content, is_test=bool(TEST_FILE.match(path)))
        if problems:
            raise GuardrailError("Rejected by guardrails:\n" + "\n".join(problems))
        target.write_text(content, encoding="utf-8")
        self.created.add(path)
        return f"Wrote {path}"

    def inspect_page(self, path: str, username: str, actions: list[dict[str, str]]) -> str:
        url = urljoin(self.base_url, path.lstrip("/"))
        self._require_same_origin(url)
        with sync_playwright() as playwright:
            playwright.selectors.set_test_id_attribute("data-test")
            browser = playwright.chromium.launch(headless=True)
            try:
                context = browser.new_context()
                if username:
                    context.add_cookies(
                        [{"name": "session-username", "value": username, "url": self.base_url}]
                    )
                page = context.new_page()
                page.set_default_timeout(ACTION_TIMEOUT_MS)
                page.goto(url)
                for action in actions:
                    locator = page.get_by_test_id(action["test_id"])
                    if action["op"] == "fill":
                        locator.fill(action["value"])
                    else:
                        locator.click()
                    self._require_same_origin(page.url)
                snapshot = {
                    "url": page.url,
                    "title": page.title(),
                    "test_ids": page.eval_on_selector_all("[data-test]", COLLECT_TEST_IDS_JS),
                    "accessibility": page.locator("body").aria_snapshot(),
                }
            finally:
                browser.close()
        return _limit(json.dumps(snapshot, indent=1))

    def run_tests(self) -> PytestRun:
        targets = sorted(path for path in self.created if TEST_FILE.match(path))
        if not targets:
            return PytestRun(5, "No test files have been created yet.")
        command = [
            sys.executable,
            "-m",
            "pytest",
            *targets,
            "-q",
            "--color=no",
            "--base-url",
            self.base_url,
            "--alluredir=.agent/allure-results",
            "--output=.agent/test-results",
        ]
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(SECRET_ENV_PREFIXES)
        }
        env["HEADLESS"] = "1"
        try:
            result = subprocess.run(  # noqa: S603 - fixed command; targets match TEST_FILE
                command,
                cwd=self.root,
                env=env,
                capture_output=True,
                text=True,
                timeout=TEST_RUN_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return PytestRun(3, f"Tests timed out after {TEST_RUN_TIMEOUT_SECONDS} seconds.")
        return PytestRun(result.returncode, _limit(result.stdout + result.stderr))

    def run_tests_tool(self) -> str:
        run = self.run_tests()
        return f"pytest exit code: {run.exit_code}\n{run.output}"

    def _require_same_origin(self, url: str) -> None:
        allowed, actual = urlsplit(self.base_url), urlsplit(url)
        if (allowed.scheme, allowed.netloc) != (actual.scheme, actual.netloc):
            raise GuardrailError(f"{url} is outside the application under test.")


def _limit(text: str) -> str:
    """Keep the end of long output, where pytest puts its failures and summary."""
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    return f"[earlier output omitted]\n{text[-MAX_TOOL_OUTPUT_CHARS:]}"
