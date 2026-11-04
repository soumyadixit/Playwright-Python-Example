"""Command line: python -m agent write | triage | eval."""

import argparse
import json
import sys
from pathlib import Path

import anthropic

from agent.evaluate import evaluate_triage, evaluate_writer
from agent.tools import DEFAULT_BASE_URL
from agent.triage import TriageError, attach_traces, parse_junit, render_markdown, triage_failures
from agent.writer import DEFAULT_MAX_COST_USD, DEFAULT_MAX_TURNS, render_pr_body, run_writer

ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = ROOT / ".agent"
EVALS_DIR = ROOT / "evals"
# A test that fails and is sent to a person is a valid result, as long as it was verified.
SUCCESS_STATUSES = {"passed", "bug_suspected"}


def write_command(args: argparse.Namespace) -> int:
    result = run_writer(
        args.scenario,
        ROOT,
        anthropic.Anthropic(),
        base_url=args.base_url,
        max_turns=args.max_turns,
        max_cost_usd=args.max_cost,
    )
    WORK_DIR.mkdir(exist_ok=True)
    (WORK_DIR / "pr_body.md").write_text(render_pr_body(args.scenario, result), encoding="utf-8")
    print(f"\nStatus: {result.status} (agent reported: {result.claimed_status or 'nothing'})")
    print(f"Summary: {result.summary}")
    print(f"Files: {', '.join(result.files) or 'none'}")
    print(f"Estimated cost: ${result.usage.cost_usd:.2f} over {result.turns} turns")
    return 0 if result.status in SUCCESS_STATUSES else 1


def triage_command(args: argparse.Namespace) -> int:
    failures = parse_junit(args.junit)
    attach_traces(failures, args.results)
    results, usage = triage_failures(failures, anthropic.Anthropic())
    report = render_markdown(results, usage) if results else "No failed tests found.\n"
    args.output.write_text(report, encoding="utf-8")
    print(report)
    return 0


def eval_command(args: argparse.Namespace) -> int:
    client = anthropic.Anthropic()
    if args.kind == "writer":
        scenarios = json.loads((EVALS_DIR / "writer_scenarios.json").read_text(encoding="utf-8"))
        report = evaluate_writer(ROOT, client, scenarios[: args.limit])
        total = report["scenarios"]
    else:
        cases = json.loads((EVALS_DIR / "triage_cases.json").read_text(encoding="utf-8"))
        report = evaluate_triage(client, cases[: args.limit])
        total = report["cases"]

    results_dir = EVALS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    stamp = report["ran_at"].replace(":", "-")
    path = results_dir / f"{args.kind}-{stamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{report['correct']}/{total} correct, estimated cost ${report['total_cost_usd']:.2f}")
    print(f"Saved to {path.relative_to(ROOT)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agent", description=__doc__)
    commands = parser.add_subparsers(required=True)

    write = commands.add_parser("write", help="Generate a test from a plain-English scenario")
    write.add_argument("scenario")
    write.add_argument("--base-url", default=DEFAULT_BASE_URL)
    write.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    write.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST_USD)
    write.set_defaults(handler=write_command)

    triage = commands.add_parser("triage", help="Classify the failures in a JUnit report")
    triage.add_argument("--junit", type=Path, default=Path("junit.xml"))
    triage.add_argument("--results", type=Path, default=Path("test-results"))
    triage.add_argument("--output", type=Path, default=Path("triage.md"))
    triage.set_defaults(handler=triage_command)

    evaluate = commands.add_parser("eval", help="Measure accuracy on known answers (costs money)")
    evaluate.add_argument("kind", choices=["writer", "triage"])
    evaluate.add_argument("--limit", type=int, default=None, help="Run only the first N items")
    evaluate.set_defaults(handler=eval_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.handler(args)
    except anthropic.AuthenticationError:
        print("Claude API login failed. Check ANTHROPIC_API_KEY.", file=sys.stderr)
    except anthropic.RateLimitError:
        print("Claude API rate limit reached. Try again shortly.", file=sys.stderr)
    except anthropic.APIStatusError as error:
        print(f"Claude API error {error.status_code}: {error.message}", file=sys.stderr)
    except anthropic.APIConnectionError:
        print("Could not reach the Claude API. Check your network.", file=sys.stderr)
    except TriageError as error:
        print(error, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
