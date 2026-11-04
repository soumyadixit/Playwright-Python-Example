"""Measure the agent against known answers. Every run makes real API calls and costs money."""

import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.triage import Failure, TriageError, triage_failures
from agent.writer import run_writer

IGNORED_WHEN_COPYING = shutil.ignore_patterns(
    ".git",
    ".venv",
    ".agent",
    "__pycache__",
    "allure-results",
    "allure-report",
    "test-results",
)


def evaluate_writer(
    root: Path,
    client: Any,
    scenarios: list[dict[str, Any]],
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run each scenario on a throwaway copy of the project and score the outcome."""
    rows = []
    for scenario in scenarios:
        log(f"\n=== {scenario['id']} ===")
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "project"
            shutil.copytree(root, copy, ignore=IGNORED_WHEN_COPYING)
            # Hide tests that already cover this behavior, so the agent can't copy the answer.
            for held_out in scenario.get("holdout", []):
                (copy / held_out).unlink()
            result = run_writer(scenario["request"], copy, client, log=log)
            generated = "\n".join(
                (copy / path).read_text(encoding="utf-8") for path in result.files
            )
        missing = [text for text in scenario["must_contain"] if text not in generated]
        rows.append(
            {
                "id": scenario["id"],
                "expected_status": scenario["expected_status"],
                "status": result.status,
                "claimed_status": result.claimed_status,
                "missing_expected_text": missing,
                "correct": result.status == scenario["expected_status"] and not missing,
                "turns": result.turns,
                "cost_usd": round(result.usage.cost_usd, 3),
            }
        )
    return {
        **_header("writer"),
        "scenarios": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "total_cost_usd": round(sum(row["cost_usd"] for row in rows), 3),
        "rows": rows,
    }


def evaluate_triage(
    client: Any,
    cases: list[dict[str, Any]],
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Classify each labeled failure on its own and compare with the expected label."""
    rows = []
    total_cost = 0.0
    for case in cases:
        failure = Failure(case["test"], case["message"], case["details"], case.get("trace", ""))
        try:
            results, usage = triage_failures([failure], client)
        except TriageError as error:
            log(f"{case['id']}: {error}")
            rows.append(
                {"id": case["id"], "expected": case["expected"], "label": "error", "correct": False}
            )
            continue
        total_cost += usage.cost_usd
        label = results[0]["label"]
        log(f"{case['id']}: expected {case['expected']}, got {label}")
        rows.append(
            {
                "id": case["id"],
                "expected": case["expected"],
                "label": label,
                "confidence": results[0]["confidence"],
                "correct": label == case["expected"],
            }
        )
    confusion = Counter(f"{row['expected']} -> {row['label']}" for row in rows)
    return {
        **_header("triage"),
        "cases": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "total_cost_usd": round(total_cost, 3),
        "confusion": dict(sorted(confusion.items())),
        "rows": rows,
    }


def _header(kind: str) -> dict[str, str]:
    return {"eval": kind, "ran_at": datetime.now(UTC).isoformat(timespec="seconds")}
