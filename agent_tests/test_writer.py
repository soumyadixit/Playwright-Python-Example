from types import SimpleNamespace

import pytest

from agent.tools import PytestRun, Toolbox
from agent.writer import decide_status, render_pr_body, run_writer

TEST_PATH = "tests/generated/cart_test.py"
TEST_CODE = "def test_cart(page, expect):\n    expect(page).to_have_title('Swag Labs')\n"


def tool_use(name, tool_input, call_id="call"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=call_id)


def response(stop_reason, *content):
    usage = SimpleNamespace(input_tokens=100, output_tokens=50)
    return SimpleNamespace(stop_reason=stop_reason, content=list(content), usage=usage)


class FakeClient:
    """Replays scripted API responses and records the requests made."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "tests/generated").mkdir(parents=True)
    (tmp_path / "src/pages").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def exit_codes(monkeypatch):
    """Replace real pytest runs with the listed exit codes."""
    codes = []

    def fake_run(_self):
        return PytestRun(codes.pop(0), "pytest output")

    monkeypatch.setattr(Toolbox, "run_tests", fake_run)
    return codes


def run(project, client, **kwargs):
    return run_writer("scenario", project, client, log=lambda _: None, **kwargs)


def write_then_finish(status="passed"):
    return [
        response("tool_use", tool_use("write_file", {"path": TEST_PATH, "content": TEST_CODE})),
        response("tool_use", tool_use("finish", {"status": status, "summary": "done"})),
    ]


@pytest.mark.parametrize(
    ("claimed", "codes", "expected"),
    [
        ("passed", [0, 0], "passed"),
        ("passed", [1, 1], "failed"),
        ("passed", [5, 5], "failed"),
        ("passed", [0, 1], "flaky"),
        ("bug_suspected", [1, 1], "bug_suspected"),
        ("bug_suspected", [0, 0], "failed"),
        ("bug_suspected", [2, 2], "failed"),
        ("bug_suspected", [1, 0], "flaky"),
    ],
)
def test_decide_status_trusts_test_runs_over_the_claim(claimed, codes, expected):
    assert decide_status(claimed, [PytestRun(code, "") for code in codes]) == expected


def test_success_is_verified_independently(project, exit_codes):
    exit_codes.extend([0, 0])
    result = run(project, FakeClient(*write_then_finish()))
    assert result.status == "passed"
    assert result.files == [TEST_PATH]
    assert (project / TEST_PATH).exists()
    assert exit_codes == []


def test_false_claim_of_success_is_caught(project, exit_codes):
    exit_codes.extend([1, 1])
    result = run(project, FakeClient(*write_then_finish()))
    assert result.status == "failed"
    assert result.claimed_status == "passed"


def test_inconsistent_runs_are_reported_as_flaky(project, exit_codes):
    exit_codes.extend([0, 1])
    assert run(project, FakeClient(*write_then_finish())).status == "flaky"


def test_guardrail_rejection_goes_back_to_the_model(project, exit_codes):
    exit_codes.extend([0, 0])
    weak = "def test_cart():\n    assert True\n"
    client = FakeClient(
        response("tool_use", tool_use("write_file", {"path": TEST_PATH, "content": weak})),
        *write_then_finish(),
    )
    result = run(project, client)
    rejection = client.requests[1]["messages"][-1]["content"][0]
    assert rejection["is_error"] is True
    assert "Rejected by guardrails" in rejection["content"]
    assert result.status == "passed"


def test_turn_limit_stops_the_loop(project):
    calls = [response("tool_use", tool_use("list_files", {})) for _ in range(5)]
    (project / "src/pages/a.py").write_text("")
    client = FakeClient(*calls)
    result = run(project, client, max_turns=3)
    assert result.status == "blocked"
    assert "3 turns" in result.summary
    assert len(client.requests) == 3


def test_budget_stops_the_loop(project):
    client = FakeClient(*[response("tool_use", tool_use("list_files", {})) for _ in range(5)])
    result = run(project, client, max_cost_usd=0.001)
    assert result.status == "blocked"
    assert "budget" in result.summary
    assert len(client.requests) == 1


def test_refusal_is_reported_without_running_tests(project):
    result = run(project, FakeClient(response("refusal")))
    assert result.status == "blocked"
    assert "declined" in result.summary


def test_blocked_claim_skips_verification(project):
    finish = tool_use("finish", {"status": "blocked", "summary": "needs a login"})
    result = run(project, FakeClient(response("tool_use", finish)))
    assert result.status == "blocked"
    assert result.summary == "needs a login"


def test_model_is_prompted_when_it_stops_without_finishing(project):
    finish = tool_use("finish", {"status": "blocked", "summary": "x"})
    client = FakeClient(response("end_turn"), response("tool_use", finish))
    run(project, client)
    assert "Call finish" in client.requests[1]["messages"][-1]["content"]


def test_every_parallel_tool_call_gets_a_result_in_one_message(project):
    finish = tool_use("finish", {"status": "blocked", "summary": "x"})
    client = FakeClient(
        response(
            "tool_use",
            tool_use("list_files", {}, "first"),
            tool_use("not_a_tool", {}, "second"),
        ),
        response("tool_use", finish),
    )
    run(project, client)
    results = client.requests[1]["messages"][-1]["content"]
    assert [item["tool_use_id"] for item in results] == ["first", "second"]
    assert results[1]["is_error"] is True


def test_requests_use_the_configured_model_and_fallback(project):
    client = FakeClient(response("refusal"))
    run(project, client)
    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["fallbacks"] == "default"
    assert request["betas"] == ["server-side-fallback-2026-07-01"]


def test_pr_body_shows_verified_status_and_asks_for_review(project, exit_codes):
    exit_codes.extend([1, 1])
    result = run(project, FakeClient(*write_then_finish("bug_suspected")))
    body = render_pr_body("Cart works", result)
    assert "`bug_suspected`" in body
    assert "review required" in body
    assert TEST_PATH in body
