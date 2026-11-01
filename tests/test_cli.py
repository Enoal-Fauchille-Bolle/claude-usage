import datetime as dt
import io
import json
import os
import sys
import time

import pytest

from claude_usage import cli

EMAIL = "someone@example.org"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake $HOME, with every path of the tool pointing inside it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_USAGE_SESSION_BUDGET", raising=False)
    monkeypatch.delenv("CLAUDE_USAGE_WEEKLY_BUDGET", raising=False)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    for name in cli.GUARD_DEFAULTS:
        monkeypatch.delenv(f"CLAUDE_USAGE_{name}", raising=False)
    state = tmp_path / ".claude" / "state"
    state.mkdir(parents=True)
    projects = tmp_path / ".claude" / "projects"
    projects.mkdir()
    monkeypatch.setattr(cli, "PROJECTS", str(projects))
    monkeypatch.setattr(cli, "STATE_DIR", str(state))
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


def status_line(home, five=28, seven=57, tokens=(100, 200, 300)):
    payload = {
        "transcript_path": str(home / "session-1.jsonl"),
        "cost": {"total_cost_usd": 3.5},
        "context_window": {
            "context_window_size": 1_000_000,
            "used_percentage": 12.6,
            "current_usage": dict(zip(("input_tokens", "cache_creation_input_tokens",
                                       "cache_read_input_tokens"), tokens)),
        },
        "rate_limits": {},
    }
    if five is not None:
        payload["rate_limits"]["five_hour"] = {"used_percentage": five, "resets_at": 1793479200}
    if seven is not None:
        payload["rate_limits"]["seven_day"] = {"used_percentage": seven, "resets_at": 1793714400}
    return payload


def run_command(monkeypatch, command, payload):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    return run_main(monkeypatch, command)


def history_lines(home):
    path = home / ".claude" / "state" / "quota-history.jsonl"
    return path.read_text().splitlines() if path.exists() else []


def context_figures(home, session="session-1"):
    text = (home / ".claude" / "state" / f"ctx-{session}.txt").read_text()
    return dict(line.split("=", 1) for line in text.splitlines())


def test_record_appends_the_documented_format(home, monkeypatch, capsys):
    assert run_command(monkeypatch, "record", status_line(home)) == 0
    assert capsys.readouterr().out == ""
    line = history_lines(home)[0]
    assert line.startswith('{"ts":')
    assert line.endswith(',"account":".claude","five":28,"seven":57,'
                         '"five_reset":1793479200,"seven_reset":1793714400}')


def test_record_appends_only_when_a_value_changes(home, monkeypatch):
    for five in (28, 28, 29):
        assert run_command(monkeypatch, "record", status_line(home, five=five)) == 0
    assert [json.loads(line)["five"] for line in history_lines(home)] == [28, 29]


def test_record_extends_an_existing_history(home, monkeypatch):
    state = home / ".claude" / "state"
    old = '{"ts":1,"account":".claude","five":1,"seven":2,"five_reset":3,"seven_reset":4}\n'
    (state / "quota-history.jsonl").write_text(old)
    (state / "quota-last-.claude.txt").write_text("28|57|1793479200|1793714400")

    assert run_command(monkeypatch, "record", status_line(home)) == 0
    assert history_lines(home) == [old.strip()]

    assert run_command(monkeypatch, "record", status_line(home, five=29)) == 0
    assert history_lines(home)[0] == old.strip()
    assert len(history_lines(home)) == 2


def test_record_writes_null_for_a_missing_window(home, monkeypatch):
    assert run_command(monkeypatch, "record", status_line(home, five=None)) == 0
    reading = json.loads(history_lines(home)[0])
    assert (reading["five"], reading["five_reset"], reading["seven"]) == (None, None, 57)


def test_record_without_quota_leaves_no_history(home, monkeypatch):
    assert run_command(monkeypatch, "record", status_line(home, five=None, seven=None)) == 0
    assert history_lines(home) == []
    assert context_figures(home)["five"] == ""


def test_record_names_the_account_after_the_config_dir(home, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / ".claude2") + "/")
    assert run_command(monkeypatch, "record", status_line(home)) == 0
    assert json.loads(history_lines(home)[0])["account"] == ".claude2"


def test_record_publishes_the_context_figures(home, monkeypatch):
    (home / "session-1.jsonl").write_text('{"timestamp":"2026-11-01T10:00:00.000Z"}\n')
    assert run_command(monkeypatch, "record", status_line(home, five=28.7)) == 0
    expiry = int(dt.datetime(2026, 11, 1, 10, tzinfo=dt.timezone.utc).timestamp()) + 3600
    assert context_figures(home) == {
        "pct": "13", "tokens": "600", "window": "1000000", "cost": "3.5",
        "five": "28", "seven": "57", "five_reset": "1793479200", "seven_reset": "1793714400",
        "cache_expiry": str(expiry),
    }


def test_record_derives_the_tokens_from_the_percentage_without_usage(home, monkeypatch):
    assert run_command(monkeypatch, "record", status_line(home, tokens=(0, 0, 0))) == 0
    assert context_figures(home)["tokens"] == "126000"
    assert context_figures(home)["cache_expiry"] == "0"


def test_record_sweeps_old_context_files_once_a_day(home, monkeypatch):
    state = home / ".claude" / "state"
    old = state / "ctx-old.txt"
    old.write_text("pct=1\n")
    eight_days_ago = time.time() - 8 * 86400
    os.utime(old, (eight_days_ago, eight_days_ago))

    assert run_command(monkeypatch, "record", status_line(home)) == 0
    assert not old.exists()
    assert (state / "ctx-session-1.txt").exists()

    old.write_text("pct=1\n")
    os.utime(old, (eight_days_ago, eight_days_ago))
    assert run_command(monkeypatch, "record", status_line(home)) == 0
    assert old.exists()


@pytest.mark.parametrize("stdin", ["not json", "[1]"])
def test_record_reports_bad_input(home, monkeypatch, capsys, stdin):
    assert run_command(monkeypatch, "record", stdin) == 1
    assert "claude-usage record" in capsys.readouterr().err
    assert history_lines(home) == []


def guard_lines(home, monkeypatch, capsys, **figures):
    state = home / ".claude" / "state"
    (state / "ctx-session-1.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in figures.items()))
    payload = {"transcript_path": str(home / "session-1.jsonl")}
    assert run_command(monkeypatch, "guard", payload) == 0
    return capsys.readouterr().out.splitlines()


def resets_in(seconds):
    return int(time.time()) + seconds


def test_guard_is_silent_below_every_threshold(home, monkeypatch, capsys):
    assert guard_lines(home, monkeypatch, capsys, tokens=1000, window=1_000_000, cost=1) == []


@pytest.mark.parametrize("tokens, name", [
    (150_000, "CONTEXT_DECISION"),
    (300_000, "CONTEXT_STOP"),
])
def test_guard_reports_the_context_threshold(home, monkeypatch, capsys, tokens, name):
    lines = guard_lines(home, monkeypatch, capsys, pct=tokens // 10_000, tokens=tokens,
                        window=1_000_000)
    assert lines[0] == f"[budget] context {tokens // 1000}k ({tokens // 10_000}%)"
    assert lines[1] == f"[budget] thresholds crossed: {name}"


def test_guard_reports_the_cost(home, monkeypatch, capsys):
    lines = guard_lines(home, monkeypatch, capsys, tokens=1000, window=1_000_000, cost=15.5)
    assert lines == ["[budget] context 1k (0%) · $15.50", "[budget] thresholds crossed: COST"]


def test_guard_ignores_a_high_quota_behind_the_pace(home, monkeypatch, capsys):
    # 73% with 20% of the window left: the pace is 80%, so 7 points of margin.
    lines = guard_lines(home, monkeypatch, capsys, tokens=1000, window=1_000_000,
                        five=73, five_reset=resets_in(3600))
    assert lines == []


def test_guard_reports_a_quota_ahead_of_the_pace(home, monkeypatch, capsys):
    # 50% with 80% of the 5-hour window left: the pace is 20%.
    lines = guard_lines(home, monkeypatch, capsys, tokens=1000, window=1_000_000,
                        five=50, five_reset=resets_in(4 * 3600))
    assert "5h 50% (pace 20%, +30)" in lines[0]
    assert lines[1] == "[budget] thresholds crossed: QUOTA_5H"


def test_guard_reports_a_quota_past_the_hard_limit_whatever_the_pace(home, monkeypatch, capsys):
    lines = guard_lines(home, monkeypatch, capsys, tokens=1000, window=1_000_000,
                        seven=95, seven_reset=resets_in(60))
    assert lines[1] == "[budget] thresholds crossed: QUOTA_7D"


def test_guard_falls_back_on_a_fixed_threshold_without_a_reset_time(home, monkeypatch, capsys):
    lines = guard_lines(home, monkeypatch, capsys, tokens=1000, window=1_000_000, five=70)
    assert lines == ["[budget] context 1k (0%) · 5h 70%", "[budget] thresholds crossed: QUOTA_5H"]


def test_guard_reports_a_cold_cache_only_on_a_big_context(home, monkeypatch, capsys):
    expired = resets_in(-60)
    assert guard_lines(home, monkeypatch, capsys, tokens=1000, window=1_000_000,
                       cache_expiry=expired) == []
    lines = guard_lines(home, monkeypatch, capsys, tokens=100_000, window=1_000_000,
                        cache_expiry=expired)
    assert f"cold cache since {dt.datetime.fromtimestamp(expired):%H:%M}" in lines[0]
    assert lines[1] == "[budget] thresholds crossed: COLD_CACHE"


def assistant_turn(tokens, sidechain=False):
    return json.dumps({"type": "assistant", "isSidechain": sidechain,
                       "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "message": {"usage": {"input_tokens": tokens,
                                             "cache_read_input_tokens": 1}}})


def test_guard_reads_the_transcript_when_the_figures_are_stale(home, monkeypatch, capsys):
    (home / "session-1.jsonl").write_text(
        assistant_turn(200_000) + "\n" + assistant_turn(500_000, sidechain=True) + "\n")
    state = home / ".claude" / "state" / "ctx-session-1.txt"
    state.write_text("tokens=900000\nwindow=1000000\nfive=99\n")
    long_ago = time.time() - 1000
    os.utime(state, (long_ago, long_ago))

    assert run_command(monkeypatch, "guard",
                       {"transcript_path": str(home / "session-1.jsonl")}) == 0
    assert capsys.readouterr().out.splitlines() == [
        "[budget] context 200k (20%)", "[budget] thresholds crossed: CONTEXT_DECISION"]


def test_guard_thresholds_come_from_the_environment(home, monkeypatch, capsys):
    monkeypatch.setenv("CLAUDE_USAGE_CONTEXT_DECISION", "500")
    lines = guard_lines(home, monkeypatch, capsys, tokens=1000, window=1_000_000)
    assert lines[1] == "[budget] thresholds crossed: CONTEXT_DECISION"


@pytest.mark.parametrize("value", ["abc", "-5", "inf", "nan"])
def test_guard_ignores_an_invalid_threshold(home, monkeypatch, capsys, value):
    monkeypatch.setenv("CLAUDE_USAGE_CONTEXT_DECISION", value)
    assert guard_lines(home, monkeypatch, capsys, tokens=1000, window=1_000_000) == []


@pytest.mark.parametrize("stdin", ["not json", "[1]", "{}"])
def test_guard_lets_the_prompt_through_on_bad_input(home, monkeypatch, capsys, stdin):
    assert run_command(monkeypatch, "guard", stdin) == 0
    assert capsys.readouterr().out == ""


def test_guard_survives_a_mangled_state_file(home, monkeypatch, capsys):
    lines = guard_lines(home, monkeypatch, capsys, tokens=200_000, window="x", five="y", cost="z")
    assert lines == ["[budget] context 200k (20%)", "[budget] thresholds crossed: CONTEXT_DECISION"]


def test_commands_ignore_an_invalid_budget_variable(home, monkeypatch):
    monkeypatch.setenv("CLAUDE_USAGE_WEEKLY_BUDGET", "abc")
    assert run_command(monkeypatch, "guard", "{}") == 0
    assert run_command(monkeypatch, "record", status_line(home)) == 0


def test_a_command_takes_no_option(home, monkeypatch):
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, "--days", "7", "record")
    assert exit_info.value.code == 2
