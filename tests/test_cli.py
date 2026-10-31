import datetime as dt
import json
import sys

import pytest

from claude_usage import cli

EMAIL = "someone@example.org"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake $HOME, with every path of the tool pointing inside it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_USAGE_SESSION_BUDGET", raising=False)
    monkeypatch.delenv("CLAUDE_USAGE_WEEKLY_BUDGET", raising=False)
    state = tmp_path / ".claude" / "state"
    state.mkdir(parents=True)
    projects = tmp_path / ".claude" / "projects"
    projects.mkdir()
    monkeypatch.setattr(cli, "PROJECTS", str(projects))
    monkeypatch.setattr(cli, "QUOTA_HISTORY", str(state / "quota-history.jsonl"))
    monkeypatch.setattr(cli, "QUOTA_GRAPH", str(state / "quota-graph.html"))
    monkeypatch.setattr(cli, "WEB_SHARE", str(state / "web-share.jsonl"))
    return tmp_path


def request(request_id="r1", model="claude-opus-5-5", **usage):
    return {
        "type": "assistant",
        "requestId": request_id,
        "sessionId": "s1",
        "cwd": "/work/demo",
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "message": {"model": model, "usage": usage},
    }


def write_transcript(home, *entries):
    folder = home / ".claude" / "projects" / "-work-demo"
    folder.mkdir(exist_ok=True)
    lines = (json.dumps(entry) for entry in entries)
    (folder / "s1.jsonl").write_text("\n".join(lines) + "\n")


def run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["claude-usage", *args])
    return cli.main()


def test_load_prices_a_request(home):
    write_transcript(home, request(input_tokens=1_000_000, output_tokens=1_000_000))
    rows = cli.load(None)
    assert len(rows) == 1
    assert rows[0]["cost"] == pytest.approx(4 + 20)
    assert rows[0]["project"] == "demo"


def test_load_counts_a_request_once_per_request_id(home):
    entry = request(output_tokens=1_000_000)
    write_transcript(home, entry, entry)
    assert len(cli.load(None)) == 1


def test_load_warns_about_an_unpriced_claude_model(home, capsys):
    write_transcript(home, request(model="claude-unknown", output_tokens=10))
    assert cli.load(None) == []
    assert "claude-unknown" in capsys.readouterr().err


def test_quota_segments_ignore_stale_and_corrupt_readings():
    records = [
        {"ts": 100, "five": 10, "five_reset": 1000},
        {"ts": 200, "five": 5, "five_reset": 1000},  # stale: lower than the last
        {"ts": 300, "five": 20, "five_reset": 1000},
        {"ts": 400, "five": 1000, "five_reset": 1000},  # corrupt percentage
        {"ts": 500, "five": 30, "five_reset": 10**9},  # reset too far ahead
    ]
    segments = cli.quota_segments(records, "five")
    assert [s["points"] for s in segments] == [[[100, 10], [300, 20]]]


def test_budgets_fall_back_on_the_constants(home):
    out = cli.budgets([], 7)
    assert (out["session"], out["weekly"]) == (cli.SESSION_BUDGET, cli.WEEKLY_BUDGET)
    assert out["session_from"] == out["weekly_from"] == "constant"


def test_budget_flags_replace_the_defaults(home, monkeypatch, capsys):
    write_transcript(home, request(output_tokens=10))
    assert run_main(monkeypatch, "--json", "--weekly-budget", "1350") == 0
    budget = json.loads(capsys.readouterr().out)["budgets"]
    assert (budget["weekly"], budget["weekly_from"]) == (1350, "set by the user")
    assert budget["session"] == cli.SESSION_BUDGET


def test_budget_environment_variable_is_read(home, monkeypatch, capsys):
    write_transcript(home, request(output_tokens=10))
    monkeypatch.setenv("CLAUDE_USAGE_SESSION_BUDGET", "200")
    assert run_main(monkeypatch, "--json") == 0
    assert json.loads(capsys.readouterr().out)["budgets"]["session"] == 200


@pytest.mark.parametrize("value", ["abc", "0", "-5", "inf", "nan"])
def test_invalid_budget_is_refused(home, monkeypatch, value):
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, f"--weekly-budget={value}")
    assert exit_info.value.code == 2


def test_report_never_shows_the_account_email(home, monkeypatch, capsys):
    (home / ".claude.json").write_text(json.dumps({"oauthAccount": {"emailAddress": EMAIL}}))
    for config_dir, session in ((".claude", "s1"), (".claude2", "s2")):
        (home / config_dir).mkdir(exist_ok=True)
        (home / config_dir / "history.jsonl").write_text(json.dumps({"sessionId": session}) + "\n")
    write_transcript(home, request(output_tokens=1_000))

    assert run_main(monkeypatch) == 0
    output = capsys.readouterr().out
    assert "BY ACCOUNT" in output
    assert ".claude2" in output
    assert EMAIL not in output


def test_graph_embeds_the_data_safely(home, monkeypatch, capsys):
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())
    reading = {"ts": now, "account": "</script>", "five": 10, "five_reset": now + 3600}
    (home / ".claude" / "state" / "quota-history.jsonl").write_text(json.dumps(reading) + "\n")
    target = home / "graph.html"

    assert run_main(monkeypatch, "--graph", str(target)) == 0
    page = target.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    assert "__DATA__" not in page
    assert '"account": "<\\/script>"' in page


def test_web_share_needs_a_percentage(home, capsys):
    assert cli.record_web(150) == 1
    assert "between 0 and 100" in capsys.readouterr().err
