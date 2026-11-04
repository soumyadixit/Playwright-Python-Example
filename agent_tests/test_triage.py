import json
import zipfile
from types import SimpleNamespace

import pytest

from agent.llm import Usage
from agent.triage import (
    Failure,
    TriageError,
    attach_traces,
    parse_junit,
    render_markdown,
    summarize_trace,
    triage_failures,
)

JUNIT = """<?xml version="1.0"?>
<testsuites><testsuite>
  <testcase classname="tests.login_test.TestLogin" name="test_ok[chromium]"/>
  <testcase classname="tests.login_test.TestLogin" name="test_bad[chromium-locked_user]">
    <failure message="AssertionError: wrong text">stack trace here</failure>
  </testcase>
  <testcase classname="tests.cart_test.TestCart" name="test_setup[chromium]">
    <error message="fixture failed">setup stack</error>
  </testcase>
</testsuite></testsuites>
"""


def write_trace(path):
    trace_events = [
        {"type": "before", "callId": "1", "method": "goto", "params": {"url": "https://a.test/"}},
        {"type": "after", "callId": "1"},
        {"type": "before", "callId": "2", "method": "click", "params": {"selector": "#buy"}},
        {"type": "after", "callId": "2", "error": {"message": "Timeout 5000ms exceeded"}},
        {"type": "console", "messageType": "error", "text": "cart.js failed"},
        {"type": "console", "messageType": "log", "text": "just a log"},
    ]
    network_events = [
        {
            "snapshot": {
                "request": {"method": "GET", "url": "https://a.test/"},
                "response": {"status": 200},
            }
        },
        {
            "snapshot": {
                "request": {"method": "POST", "url": "https://a.test/cart"},
                "response": {"status": 500},
            }
        },
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("trace.trace", "\n".join(map(json.dumps, trace_events)) + "\nnot json\n")
        archive.writestr("trace.network", "\n".join(map(json.dumps, network_events)))


def fake_client(payload, stop_reason="end_turn"):
    message = SimpleNamespace(
        stop_reason=stop_reason,
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=json.dumps(payload)),
        ],
        usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
    )
    client = SimpleNamespace(requests=[])

    def create(**kwargs):
        client.requests.append(kwargs)
        return message

    client.beta = SimpleNamespace(messages=SimpleNamespace(create=create))
    return client


def result_for(test, label="test_bug"):
    return {
        "test": test,
        "label": label,
        "confidence": "high",
        "evidence": "quoted detail",
        "next_step": "fix the locator",
    }


def test_parse_junit_finds_failures_and_errors_only(tmp_path):
    report = tmp_path / "junit.xml"
    report.write_text(JUNIT)
    failures = parse_junit(report)
    assert [failure.test for failure in failures] == [
        "tests.login_test.TestLogin::test_bad[chromium-locked_user]",
        "tests.cart_test.TestCart::test_setup[chromium]",
    ]
    assert failures[0].message == "AssertionError: wrong text"
    assert failures[0].details == "stack trace here"


def test_unreadable_report_is_a_clear_error(tmp_path):
    with pytest.raises(TriageError):
        parse_junit(tmp_path / "missing.xml")
    broken = tmp_path / "junit.xml"
    broken.write_text("")
    with pytest.raises(TriageError):
        parse_junit(broken)


def test_summarize_trace_keeps_the_useful_signals(tmp_path):
    path = tmp_path / "trace.zip"
    write_trace(path)
    summary = summarize_trace(path)
    assert "click #buy: Timeout 5000ms exceeded" in summary
    assert "cart.js failed" in summary
    assert "500 POST https://a.test/cart" in summary
    assert "just a log" not in summary
    assert "200 GET" not in summary


def test_unreadable_trace_does_not_stop_triage(tmp_path):
    path = tmp_path / "trace.zip"
    path.write_text("not a zip")
    assert summarize_trace(path) == ""


def test_traces_attach_to_the_matching_parametrized_test(tmp_path):
    folder = tmp_path / "tests-login-test-py-testlogin-test-bad-chromium-locked-user"
    folder.mkdir()
    write_trace(folder / "trace.zip")
    matching = Failure("tests.login_test.TestLogin::test_bad[chromium-locked_user]", "", "")
    other = Failure("tests.login_test.TestLogin::test_bad[chromium-other_user]", "", "")
    attach_traces([matching, other], tmp_path)
    assert "click #buy" in matching.trace
    assert other.trace == ""


def test_missing_results_folder_is_fine(tmp_path):
    attach_traces([Failure("a::b", "", "")], tmp_path / "missing")


def test_triage_returns_results_in_failure_order():
    failures = [Failure("a::one", "m", "d"), Failure("a::two", "m", "d")]
    client = fake_client({"results": [result_for("a::two"), result_for("a::one", "flaky")]})
    results, usage = triage_failures(failures, client)
    assert [item["label"] for item in results] == ["flaky", "test_bug"]
    assert usage.input_tokens == 1000
    request = client.requests[0]
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "a::one" in request["messages"][0]["content"]


def test_unclassified_test_is_flagged_for_a_person():
    client = fake_client({"results": []})
    results, _ = triage_failures([Failure("a::one", "m", "d")], client)
    assert results[0]["label"] == "needs_review"


def test_no_failures_means_no_api_call():
    client = fake_client({"results": []})
    assert triage_failures([], client)[0] == []
    assert client.requests == []


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_incomplete_response_raises(stop_reason):
    client = fake_client({"results": []}, stop_reason)
    with pytest.raises(TriageError):
        triage_failures([Failure("a::one", "m", "d")], client)


def test_markdown_says_results_are_suggestions():
    report = render_markdown([result_for("a::one", "product_bug")], Usage())
    assert "`a::one`" in report
    assert "**product_bug** (high confidence)" in report
    assert "human to confirm" in report
