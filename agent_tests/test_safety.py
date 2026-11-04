import pytest

from agent.safety import GuardrailError, resolve_read_path, resolve_write_path, validate_python

GOOD_TEST = """
import pytest
from playwright.sync_api import Page, expect

from src.enums.User import User


class TestCart:
    @pytest.mark.parametrize("browser_context_args", [User.STANDARD_USER], indirect=True)
    def test_badge(self, browser_context_args, page: Page) -> None:
        page.get_by_test_id("add-to-cart").click()
        expect(page.get_by_test_id("badge")).to_have_text("1")
"""


def test_good_test_passes():
    assert validate_python(GOOD_TEST, is_test=True) == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def test_a(page):\n    page.goto('/')\n", "has no assertion"),
        ("def test_a():\n    assert True\n", "asserting a constant"),
        ("x = 1\n", "no test_ functions"),
        ("def test_a(:\n", "Syntax error"),
        ("import os\ndef test_a(x):\n    assert x\n", "import of os"),
        ("from subprocess import run\ndef test_a(x):\n    assert x\n", "import of subprocess"),
        ("from . import y\ndef test_a(x):\n    assert x\n", "import of ."),
        ("def test_a(x):\n    assert x\n    eval('1')\n", "eval()"),
        ("def test_a(p, x):\n    p.wait_for_timeout(500)\n    assert x\n", "fixed wait"),
        ("import pytest\ndef test_a(x):\n    pytest.skip('no')\n    assert x\n", "pytest.skip"),
        (
            "import pytest\n@pytest.mark.xfail\ndef test_a(x):\n    assert x\n",
            "pytest.mark.xfail",
        ),
        (
            "def test_a(x, expect):\n    try:\n        expect(x).to_be_visible()\n"
            "    except Exception:\n        pass\n",
            "try/except",
        ),
        (
            "import re\ndef test_a(p, expect):\n    expect(p).to_have_url(re.compile('.*'))\n",
            "matches anything",
        ),
        ("def test_a(x):\n    assert x.__class__\n", "dunder"),
    ],
)
def test_weak_or_unsafe_tests_are_rejected(source, expected):
    problems = validate_python(source, is_test=True)
    assert any(expected in problem for problem in problems), problems


def test_expect_without_a_matcher_call_is_not_an_assertion():
    source = "def test_a(p, expect):\n    expect(p).to_be_visible\n"
    assert any("no assertion" in problem for problem in validate_python(source, is_test=True))


def test_page_objects_do_not_need_tests_but_keep_the_other_rules():
    assert validate_python("class CartPage:\n    pass\n", is_test=False) == []
    assert validate_python("import os\n", is_test=False)


@pytest.mark.parametrize(
    "path",
    [
        "tests/login_test.py",
        "tests/conftest.py",
        "tests/generated/../../agent/tools.py",
        "tests/generated/nested/a_test.py",
        "tests/generated/not_a_test_name.py",
        "src/pages/../enums/User.py",
        ".github/workflows/devRun.yml",
        "pyproject.toml",
        "/etc/passwd",
    ],
)
def test_writes_outside_agent_folders_are_blocked(tmp_path, path):
    with pytest.raises(GuardrailError):
        resolve_write_path(tmp_path, path, set())


def test_existing_files_are_read_only_unless_created_in_this_run(tmp_path):
    (tmp_path / "src/pages").mkdir(parents=True)
    (tmp_path / "src/pages/login_page.py").write_text("x = 1")
    path = "src/pages/login_page.py"
    with pytest.raises(GuardrailError):
        resolve_write_path(tmp_path, path, set())
    assert resolve_write_path(tmp_path, path, {path}).name == "login_page.py"


def test_symlink_cannot_escape_the_project(tmp_path):
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    (project / "tests/generated").symlink_to(tmp_path / "outside")
    with pytest.raises(GuardrailError):
        resolve_write_path(project, "tests/generated/a_test.py", set())


@pytest.mark.parametrize(
    "path", [".env", ".git/config", "uv.lock", "src/../.env", "src/../../outside.py"]
)
def test_reads_are_limited_to_project_code(tmp_path, path):
    with pytest.raises(GuardrailError):
        resolve_read_path(tmp_path, path)
