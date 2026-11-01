"""Analyse local Claude Code transcripts and report on plan-limit consumption.

Claude Code stores one JSONL transcript per session under ~/.claude/projects/.
Every assistant message carries a `usage` object with the exact token counts the
API billed for that request. This script replays those counts to answer three
questions the built-in /usage command cannot:

  1. How close am I to the weekly and 5-hour plan limits, day by day?
  2. Which sessions are burning budget on re-read context rather than on work?
  3. What did stale sessions cost me in cache misses?

Everything is expressed in "API-list-price equivalent" ($). That is NOT what a
subscription bills -- the plan is a flat fee -- but it is the unit Anthropic
charges usage credits in, and it tracks plan-limit consumption closely enough to
be useful. The weekly and 5-hour budgets are measured from the /usage
percentages the status line records, with the claude.ai share given by --web;
two constants below stand in when that history is too thin. --session-budget and
--weekly-budget replace both, as do CLAUDE_USAGE_SESSION_BUDGET and
CLAUDE_USAGE_WEEKLY_BUDGET.

Usage:
    claude-usage                 # last 28 days
    claude-usage --days 7        # shorter window
    claude-usage --sessions 20   # show more sessions in the session table
    claude-usage --json          # machine-readable output
    claude-usage --graph         # HTML page: official 5-hour/weekly quota, spend by project/model,
                                 # every figure following its 24 h / 3 days / 7 days / All filter
                                 # or a range dragged across the chart
    claude-usage --graph out.html
    claude-usage --web 22        # claude.ai took 22% of this week's usage
    claude-usage --weekly-budget 1350   # a plan five times larger than Pro
    claude-usage record          # status line: save the figures read from stdin
    claude-usage guard           # UserPromptSubmit hook: warn when a threshold is crossed
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import datetime as dt
import glob
import importlib.resources
import json
import os
import re
import statistics
import sys
import time

# --- Calibration -----------------------------------------------------------
# The budgets are measured from the quota history when it allows (see
# measured_budgets()); these constants are the fallback. Measured on 2026-10-06 on a Pro
# plan, one account, Opus 5.5 and Sonnet 5.5:
#   week of 2026-09-29: 63% used, 78% of it Claude Code (claude.ai usage page)
#                       <-> $132 of equivalent spend          -> ~$270 per week
#   5-hour windows at 50% or more: $35-44 per 100% in the ones with the least
#                       claude.ai in them                     -> ~$40 per 5h
# The $330 and $64 measured in September, on two accounts and older models, no
# longer match. claude.ai shares the same limit and is invisible in the
# transcripts, hence the share recorded with --web.
SESSION_BUDGET = 40.0       # $ equivalent per 5-hour window, one Pro seat
WEEKLY_BUDGET = 270.0       # $ equivalent per week, one Pro seat

# A window counts towards a measured budget only when it reached this
# percentage: below it, the 1% rounding of /usage weighs too much. A week only
# needs its Claude Code part to reach it, since its claude.ai share is known.
CALIBRATE_MIN_PCT = {"five": 50, "seven": 20}

# Anthropic list prices, $ per million tokens (input, output, cache-read
# multiplier). Most models read the cache at 0.1x input; the newest read it
# cheaper. Source: https://platform.claude.com/docs/en/about-claude/pricing
PRICES = {
    "claude-fable-5-1": (10, 50, 0.025),
    "claude-opus-5-5": (4, 20, 0.05),
    "claude-opus-5": (5, 25, 0.1),
    "claude-opus-4-8": (5, 25, 0.1),
    "claude-opus-4-7": (5, 25, 0.1),
    "claude-opus-4-6": (5, 25, 0.1),
    "claude-sonnet-5-5": (2, 10, 0.1),
    "claude-sonnet-5": (2, 10, 0.1),
    "claude-sonnet-4-6": (3, 15, 0.1),
    "claude-haiku-4-5-20251001": (1, 5, 0.1),
}

# Cache writes cost more than fresh input.
CACHE_WRITE_5M = 1.25
CACHE_WRITE_1H = 2.0

# A request sent with more context than this is flagged as "long context".
LONG_CONTEXT = 150_000
HUGE_CONTEXT = 300_000

# A gap longer than the 1-hour cache lifetime means the next request re-writes
# the whole context at cache-write price instead of reading it at 0.1x.
CACHE_LIFETIME_H = 1.0

PROJECTS = os.path.expanduser("~/.claude/projects")

STATE_DIR = os.path.expanduser("~/.claude/state")

# claude-usage record appends one reading of the official /usage percentages here
# each time they change (format in the README).
QUOTA_HISTORY = os.path.join(STATE_DIR, "quota-history.jsonl")
QUOTA_GRAPH = os.path.join(STATE_DIR, "quota-graph.html")
QUOTA_SPAN = {"five": 5 * 3600, "seven": 7 * 86400}  # window length, seconds
# claude-usage --web appends the claude.ai share of each week here, as read on
# the "usage this week by product" panel of claude.ai.
WEB_SHARE = os.path.join(STATE_DIR, "web-share.jsonl")

# What `claude-usage guard` compares against, each replaced by the environment
# variable CLAUDE_USAGE_<name>.
GUARD_DEFAULTS = {
    "CONTEXT_DECISION": LONG_CONTEXT,  # tokens: same task, /compact; new task, /clear
    "CONTEXT_STOP": HUGE_CONTEXT,      # tokens
    "COST_WARN": 15,                   # $ in one conversation
    "PACE_GAP": 10,                    # points ahead of the share of the window elapsed
    "QUOTA_HARD": 90,                  # %: past this, the pace no longer matters
    "FIVE_WARN": 70,                   # % of the 5-hour window, when its reset time is unknown
    "SEVEN_WARN": 80,                  # % of the week, when its reset time is unknown
    "CACHE_MIN": 100_000,              # tokens: a cold cache only matters on a big context
    "CACHE_TTL": CACHE_LIFETIME_H * 3600,  # seconds; 300 when drawing on usage credits
    "STATE_MAX_AGE": 900,              # seconds: older figures from the status line are stale
}

# A session leaves its ctx-<session>.txt behind when it ends.
CONTEXT_FILE_MAX_AGE = 7 * 86400
SWEEP_EVERY = 86400

# Projects that get their own colour in the per-day chart. The categorical
# palette has eight slots; the eighth is "Other".
STACK_PROJECTS = 7


def accounts() -> list[tuple[str, set[str]]]:
    """Map each Claude config directory to the session IDs it produced.

    Running two subscriptions means two CLAUDE_CONFIG_DIRs. Their `projects/`
    directories are often symlinked together, so the transcripts are pooled and
    carry no account marker -- but each config dir keeps its own history.jsonl,
    which records the session IDs that dir has seen. That is the only local
    signal that separates the accounts.

    history.jsonl is capped, so older sessions fall out of it and end up
    unattributed. Treat the split as a lower bound per account, never as an
    exact partition.
    """
    found = []
    for config_dir in sorted(glob.glob(os.path.expanduser("~/.claude*"))):
        history = os.path.join(config_dir, "history.jsonl")
        if not os.path.isfile(history):
            continue

        # The directory name, not the account's e-mail: reports get pasted around.
        label = os.path.basename(config_dir)

        ids = set()
        with open(history, errors="replace") as fh:
            for line in fh:
                try:
                    ids.add(json.loads(line)["sessionId"])
                except (json.JSONDecodeError, KeyError):
                    continue
        if ids:
            found.append((label, ids))
    return found


def load(days: int | None) -> list[dict]:
    """Replay every assistant request from the transcripts into a flat list.

    Deduplicated on requestId: a single assistant turn is written to the
    transcript once per content block, so counting lines would double-count.
    Requests from a Claude model missing in PRICES are left out and reported
    on stderr, so a new model shows up as a warning rather than as a quiet dip.
    `days` of None keeps every request.
    """
    cutoff = None if days is None else dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    seen: dict[str, dict] = {}
    unpriced: collections.Counter[str] = collections.Counter()
    cwds: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)

    # Sub-agents (Agent tool, background tasks) write their own transcripts next
    # to the session, in <project>/<session-id>/subagents/. They are billed like
    # any other request but carry the parent's sessionId, so they fold into it.
    paths = (glob.glob(os.path.join(PROJECTS, "*", "*.jsonl"))
             + glob.glob(os.path.join(PROJECTS, "*", "*", "subagents", "*.jsonl")))

    for path in paths:
        project = os.path.relpath(path, PROJECTS).split(os.sep)[0]
        with open(path, errors="replace") as fh:
            for line in fh:
                if '"usage"' not in line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") != "assistant":
                    continue
                if entry.get("cwd"):
                    cwds[project][entry["cwd"]] += 1
                message = entry.get("message") or {}
                usage = message.get("usage") or {}
                if not usage:
                    continue
                key = entry.get("requestId") or message.get("id") or entry.get("uuid")
                if key in seen:
                    continue
                when = dt.datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
                if cutoff and when < cutoff:
                    continue

                created = usage.get("cache_creation") or {}
                write_5m = created.get("ephemeral_5m_input_tokens", 0) or 0
                write_1h = created.get("ephemeral_1h_input_tokens", 0) or 0
                write_total = usage.get("cache_creation_input_tokens", 0) or 0
                if not (write_5m or write_1h):
                    write_5m = write_total

                model = message.get("model")
                inp = usage.get("input_tokens", 0) or 0
                read = usage.get("cache_read_input_tokens", 0) or 0
                out = usage.get("output_tokens", 0) or 0

                price = PRICES.get(model)
                if price is None:
                    # Only Claude models draw on the subscription. Others reach the
                    # transcripts through a proxy (NVIDIA NIM), and "<synthetic>"
                    # marks local messages that were never sent.
                    if model and model.startswith("claude-"):
                        unpriced[model] += 1
                    continue
                rate_in, rate_out, cache_read = price
                cost = (
                    inp * rate_in
                    + write_5m * rate_in * CACHE_WRITE_5M
                    + write_1h * rate_in * CACHE_WRITE_1H
                    + read * rate_in * cache_read
                    + out * rate_out
                ) / 1e6

                seen[key] = {
                    "t": when.astimezone(),
                    "project": project,
                    "session": entry.get("sessionId"),
                    "model": model,
                    "out": out,
                    "write": write_5m + write_1h,
                    "write_cost": (write_5m * rate_in * CACHE_WRITE_5M
                                   + write_1h * rate_in * CACHE_WRITE_1H) / 1e6,
                    "context": inp + read + write_total,
                    "cost": cost,
                }

    if unpriced:
        missing = ", ".join(f"{m} ({n} requests)" for m, n in unpriced.most_common())
        print(f"warning: no price for {missing}; left out of every total. "
              "Add them to PRICES.", file=sys.stderr)

    names = project_names(cwds)
    for r in seen.values():
        r["project"], r["path"] = names.get(r["project"], (r["project"], r["project"]))

    return sorted(seen.values(), key=lambda r: r["t"])


def project_names(cwds: dict[str, collections.Counter[str]]) -> dict[str, tuple[str, str]]:
    """Map each transcript folder to a readable (name, path) pair.

    The folder name is the launch directory with every "/" turned into "-", which
    cannot be decoded (Astra-ops/infra and Astra/ops-infra encode alike), so the
    `cwd` recorded on each message is the source. A session can `cd` around; the
    most frequent cwd is the project. A project nested in another is named after
    it ("Roots-SMP/web"), and two projects that would share a name keep their path.
    """
    home = os.path.expanduser("~")
    paths = {folder: counts.most_common(1)[0][0] for folder, counts in cwds.items()}

    def name(path: str) -> str:
        parents = [p for p in paths.values() if path.startswith(p.rstrip(os.sep) + os.sep)]
        if parents:
            parent = max(parents, key=len)
            return f"{name(parent)}/{os.path.relpath(path, parent)}"
        return os.path.basename(path.rstrip(os.sep)) or path

    def tilde(path: str) -> str:
        return "~" + path[len(home):] if path == home or path.startswith(home + os.sep) else path

    names = {folder: name(path) for folder, path in paths.items()}
    clashes = {n for n, count in collections.Counter(names.values()).items() if count > 1}
    return {folder: (tilde(paths[folder]) if names[folder] in clashes else names[folder],
                     tilde(paths[folder]))
            for folder in paths}


def weekly(rows: list[dict]) -> list[tuple[dt.date, float]]:
    buckets: dict[dt.date, float] = collections.defaultdict(float)
    for r in rows:
        monday = r["t"].date() - dt.timedelta(days=r["t"].weekday())
        buckets[monday] += r["cost"]
    return sorted(buckets.items())


def daily_peaks(rows: list[dict]) -> list[tuple[dt.date, float, float]]:
    """Largest 5-hour rolling window per day, plus that day's total.

    The 5-hour limit is what actually stops you mid-session, so the peak window
    matters more than the daily total.
    """
    stamps = [r["t"] for r in rows]
    running = [0.0]
    for r in rows:
        running.append(running[-1] + r["cost"])

    peak: dict[dt.date, float] = collections.defaultdict(float)
    total: dict[dt.date, float] = collections.defaultdict(float)
    for i, r in enumerate(rows):
        day = r["t"].date()
        end = bisect.bisect_right(stamps, r["t"] + dt.timedelta(hours=5))
        peak[day] = max(peak[day], running[end] - running[i])
        total[day] += r["cost"]
    return [(d, peak[d], total[d]) for d in sorted(peak)]


def cold_rewrite(before: dict, after: dict) -> bool:
    """Whether `after` re-writes the context into a cache that went cold while idle."""
    idle = (after["t"] - before["t"]).total_seconds() / 3600
    return idle > CACHE_LIFETIME_H and after["write"] > 50_000


def sessions(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        grouped[r["session"]].append(r)

    out = []
    for sid, rs in grouped.items():
        stale = 0.0
        gaps = 0
        for before, after in zip(rs, rs[1:]):
            if cold_rewrite(before, after):
                gaps += 1
                stale += after["write_cost"]
        out.append({
            "id": sid,
            "project": rs[0]["project"],
            "path": rs[0]["path"],
            "start": rs[0]["t"],
            "hours": (rs[-1]["t"] - rs[0]["t"]).total_seconds() / 3600,
            "requests": len(rs),
            "max_context": max(r["context"] for r in rs),
            "cost": sum(r["cost"] for r in rs),
            "long_cost": sum(r["cost"] for r in rs if r["context"] > LONG_CONTEXT),
            "huge_cost": sum(r["cost"] for r in rs if r["context"] > HUGE_CONTEXT),
            "stale_gaps": gaps,
            "stale_cost": stale,
        })
    return sorted(out, key=lambda s: -s["cost"])


def model_label(model: str) -> str:
    """"claude-opus-5-5" -> "Opus 5.5"; drops the date suffix of pinned IDs."""
    parts = [p for p in model.removeprefix("claude-").split("-")
             if not (p.isdigit() and len(p) == 8)]
    return " ".join([parts[0].capitalize(), ".".join(parts[1:])]).strip()


def spend_data(rows: list[dict]) -> dict:
    """Every request, compact, for the spend half of the HTML page.

    The page adds them up itself, so that its time range filter can recut the
    spend. A request is [time, project, model, session, cost, cold-cache rewrite
    cost, 1 if over LONG_CONTEXT], the project and model being indexes into the
    two ranked lists, and the session an index of its own.
    """
    by_project: collections.Counter[str] = collections.Counter()
    by_model: collections.Counter[str] = collections.Counter()
    paths: dict[str, str] = {}
    for r in rows:
        by_project[r["project"]] += r["cost"]
        by_model[model_label(r["model"])] += r["cost"]
        paths.setdefault(r["project"], r["path"])
    ranked = [name for name, _ in by_project.most_common()]
    project_slot = {name: i for i, name in enumerate(ranked)}
    model_slot = {name: i for i, (name, _) in enumerate(by_model.most_common())}

    session_slot: dict[str, int] = {}
    previous: dict[str, dict] = {}
    requests = []
    for r in rows:
        before = previous.get(r["session"])
        previous[r["session"]] = r
        cold = before is not None and cold_rewrite(before, r)
        requests.append([
            int(r["t"].timestamp()),
            project_slot[r["project"]],
            model_slot[model_label(r["model"])],
            session_slot.setdefault(r["session"], len(session_slot)),
            round(r["cost"], 5),
            round(r["write_cost"], 5) if cold else 0,
            int(r["context"] > LONG_CONTEXT),
        ])

    return {
        "projects": [{"name": name, "path": paths[name]} for name in ranked],
        "colours": STACK_PROJECTS,
        "models": list(model_slot),
        "requests": requests,
    }


def report(rows: list[dict], days: int, n_sessions: int, budget: dict) -> None:
    if not rows:
        print("No transcripts found in the requested window.", file=sys.stderr)
        return

    total = sum(r["cost"] for r in rows)
    long_cost = sum(r["cost"] for r in rows if r["context"] > LONG_CONTEXT)
    huge_cost = sum(r["cost"] for r in rows if r["context"] > HUGE_CONTEXT)
    sess = sessions(rows)
    stale = sum(s["stale_cost"] for s in sess)

    print(f"\n  {len(rows):,} requests over {days} days — ${total:,.0f} equivalent\n")

    print(f"  WEEKLY  (one Pro seat ≈ ${budget['weekly']:,.0f}/week, {budget['weekly_from']})")
    if budget["weekly_from"] == "constant":
        print("    record this week's claude.ai share with: claude-usage --web <percent>")
    for monday, spend in weekly(rows):
        pct = spend / budget["weekly"] * 100
        bar = "█" * min(40, round(pct / 2.5))
        print(f"    {monday}  ${spend:7.0f}  {pct:5.1f}%  {bar}")

    print(f"\n  DAILY 5-HOUR PEAK  (limit ≈ ${budget['session']:.0f}, {budget['session_from']})")
    over = 0
    for day, peak, day_total in daily_peaks(rows):
        hit = peak > budget["session"]
        over += hit
        bar = "█" * min(30, round(peak / budget["session"] * 20))
        print(f"    {day}  peak ${peak:6.1f}  day ${day_total:6.1f}  {bar}"
              f"{'  <-- would hit the wall' if hit else ''}")
    print(f"    -> {over} day(s) above the 5-hour limit")

    print("\n  WHERE THE MONEY GOES")
    print(f"    context > {LONG_CONTEXT // 1000}k tokens   ${long_cost:7.0f}  {long_cost / total * 100:4.0f}%"
          "   <- /compact or /clear earlier")
    print(f"    context > {HUGE_CONTEXT // 1000}k tokens   ${huge_cost:7.0f}  {huge_cost / total * 100:4.0f}%"
          "   <- should never get here")
    print(f"    cache miss after idle  ${stale:7.0f}  {stale / total * 100:4.0f}%"
          "   <- /clear before walking away")

    known = accounts()
    if len(known) > 1:
        # A session resumed from both config dirs appears in both histories --
        # the symlinked projects/ directory makes that possible -- so count a
        # session for an account only when exactly one history claims it.
        shared: dict[str, int] = collections.Counter()
        for _, ids in known:
            for sid in ids:
                shared[sid] += 1

        print("\n  BY ACCOUNT  (from each config dir's history.jsonl — a lower bound)")
        buckets = {label: 0.0 for label, _ in known}
        both = unknown = 0.0
        for r in rows:
            sid = r["session"]
            if shared.get(sid, 0) > 1:
                both += r["cost"]
                continue
            for label, ids in known:
                if sid in ids:
                    buckets[label] += r["cost"]
                    break
            else:
                unknown += r["cost"]

        for label, spend in buckets.items():
            print(f"    {label:<38} ${spend:7.0f}  {spend / total * 100:4.0f}%")
        if both:
            print(f"    {'(resumed from both config dirs)':<38} ${both:7.0f}  {both / total * 100:4.0f}%")
        print(f"    {'(unattributed — history.jsonl is capped)':<38} ${unknown:7.0f}  {unknown / total * 100:4.0f}%")

    print(f"\n  TOP SESSIONS  (a session over ~6h or ~300k context is a /clear you skipped)")
    print(f"    {'session':<10}{'started':<14}{'hours':>7}{'reqs':>6}{'ctx max':>9}"
          f"{'cost':>8}{'>150k':>8}{'idle$':>7}  project")
    for s in sess[:n_sessions]:
        print(f"    {s['id'][:8]:<10}{s['start']:%m-%d %H:%M}  {s['hours']:6.1f}h{s['requests']:6d}"
              f"{s['max_context'] / 1000:8.0f}k{s['cost']:8.1f}{s['long_cost']:8.1f}"
              f"{s['stale_cost']:7.1f}  {s['project']}")
    print()


def quota_segments(records: list[dict], key: str) -> list[dict]:
    """Group the recorded readings of one limit into one segment per window.

    `key` is "five" or "seven". Several Claude Code sessions can run at once and
    each writes its own, sometimes stale, view of the percentage, and a few lines
    are corrupt (a reset time landing in the percentage field). Inside one window
    /usage only ever goes up, so a lower reading is stale and is ignored; a
    reading is dropped outright when its percentage is outside 0-100 or its reset
    time is not within one window length ahead of it.

    `points` are the [timestamp, percent] readings where the value rose; `end` is
    the time of the last reading, so nothing is drawn past what was recorded.
    """
    span = QUOTA_SPAN[key]
    windows: dict[int, dict] = {}
    for rec in records:
        ts, pct, reset = rec.get("ts"), rec.get(key), rec.get(f"{key}_reset")
        if not all(isinstance(v, (int, float)) for v in (ts, pct, reset)):
            continue
        if not (0 <= pct <= 100 and 0 < reset - ts <= span + 60):
            continue
        window = windows.setdefault(reset, {"reset": reset, "points": [], "end": ts})
        window["end"] = ts
        if window["points"] and pct <= window["points"][-1][1]:
            continue
        window["points"].append([ts, pct])
    return sorted(windows.values(), key=lambda w: w["reset"])


def load_quota() -> tuple[str, list[dict]]:
    """Read the quota history, keeping only the most recently recorded account."""
    records = []
    with open(QUOTA_HISTORY, errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and isinstance(rec.get("ts"), (int, float)):
                records.append(rec)
    records.sort(key=lambda r: r["ts"])
    account = records[-1].get("account", "") if records else ""
    return account, [r for r in records if r.get("account", "") == account]


def load_web_shares(account: str) -> dict[int, float]:
    """claude.ai share of each week, keyed by the week's reset time. A later
    entry for the same week replaces an earlier one, so a typo is fixed by
    recording again."""
    shares: dict[int, float] = {}
    if not os.path.isfile(WEB_SHARE):
        return shares
    with open(WEB_SHARE, errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("account", "") == account \
                    and isinstance(rec.get("seven_reset"), int) \
                    and isinstance(rec.get("web"), (int, float)):
                shares[rec["seven_reset"]] = rec["web"]
    return shares


def record_web(pct: float) -> int:
    """Append the claude.ai share of the week in progress to WEB_SHARE."""
    if not 0 <= pct <= 100:
        print("--web takes a percentage between 0 and 100.", file=sys.stderr)
        return 1
    if not os.path.isfile(QUOTA_HISTORY):
        print(f"{QUOTA_HISTORY} not found: `claude-usage record` writes it from the status line.",
              file=sys.stderr)
        return 1
    account, records = load_quota()
    weeks = quota_segments(records, "seven")
    if not weeks:
        print("No weekly quota reading recorded yet: the week in progress is unknown.",
              file=sys.stderr)
        return 1
    # The weekly reset falls at the same time every week, so a week that began
    # after the last reading is found by stepping forward from it.
    reset, now = weeks[-1]["reset"], time.time()
    while reset <= now:
        reset += QUOTA_SPAN["seven"]
    os.makedirs(os.path.dirname(WEB_SHARE), exist_ok=True)
    with open(WEB_SHARE, "a") as fh:
        fh.write(json.dumps({"ts": int(now), "account": account,
                             "seven_reset": int(reset), "web": pct}) + "\n")
    print(f"  recorded: week ending {dt.datetime.fromtimestamp(reset):%Y-%m-%d %H:%M}, "
          f"claude.ai {pct:g}%")
    return 0


def budgets(rows: list[dict], days: int, session: float | None = None,
            weekly_budget: float | None = None) -> dict:
    """The measured budgets, replaced by whichever of them the user set."""
    out = measured_budgets(rows, days)
    if session is not None:
        out["session"], out["session_from"] = session, "set by the user"
    if weekly_budget is not None:
        out["weekly"], out["weekly_from"] = weekly_budget, "set by the user"
    return out


def measured_budgets(rows: list[dict], days: int) -> dict:
    """$ equivalent of a full week and of a full 5-hour window, measured from the
    quota history, or the constants when it does not allow it.

    Weekly: Claude Code spend over a week divided by the part of the week's
    percentage that was Claude Code, which only the recorded claude.ai share
    gives; the median over such weeks.
    5-hour: claude.ai raises the percentage without leaving any spend here, so
    the windows with the most $ per % are the ones with the least claude.ai in
    them; the median of the top quarter of windows that reached
    CALIBRATE_MIN_PCT. Only windows inside the --days span count.
    """
    out = {"weekly": WEEKLY_BUDGET, "weekly_from": "constant",
           "session": SESSION_BUDGET, "session_from": "constant"}
    if not rows or not os.path.isfile(QUOTA_HISTORY):
        return out

    stamps = [r["t"].timestamp() for r in rows]
    running = [0.0]
    for r in rows:
        running.append(running[-1] + r["cost"])

    def spend(start: float, end: float) -> float:
        return running[bisect.bisect_left(stamps, end)] - running[bisect.bisect_left(stamps, start)]

    cutoff = time.time() - days * 86400
    account, records = load_quota()
    shares = load_web_shares(account)

    weekly = []
    for w in quota_segments(records, "seven"):
        start = w["reset"] - QUOTA_SPAN["seven"]
        if w["reset"] not in shares or not w["points"] or start < cutoff:
            continue
        code_pct = w["points"][-1][1] * (100 - shares[w["reset"]]) / 100
        if code_pct >= CALIBRATE_MIN_PCT["seven"]:
            weekly.append(spend(start, w["reset"]) / code_pct * 100)
    if weekly:
        out["weekly"] = statistics.median(weekly)
        out["weekly_from"] = f"{len(weekly)} week(s) with their claude.ai share"

    ratios = []
    for w in quota_segments(records, "five"):
        start = w["reset"] - QUOTA_SPAN["five"]
        if not w["points"] or start < cutoff:
            continue
        peak = w["points"][-1][1]
        if peak >= CALIBRATE_MIN_PCT["five"]:
            ratios.append(spend(start, w["reset"]) / peak * 100)
    if len(ratios) >= 4:
        top = sorted(ratios)[-(len(ratios) // 4):]
        out["session"] = statistics.median(top)
        out["session_from"] = (f"top quarter of {len(ratios)} five-hour windows "
                               f"at {CALIBRATE_MIN_PCT['five']}%+")
    return out


def write_graph(path: str) -> int:
    """Render the quota history and the Claude Code spend as a self-contained HTML
    page (no external assets). The spend covers every transcript, since the page's
    time range filter cuts it down."""
    if not os.path.isfile(QUOTA_HISTORY):
        print(f"{QUOTA_HISTORY} not found: `claude-usage record` writes it from the status line.",
              file=sys.stderr)
        return 1

    account, records = load_quota()
    five = quota_segments(records, "five")
    seven = quota_segments(records, "seven")
    if not (five or seven):
        print("No usable quota readings recorded yet.", file=sys.stderr)
        return 1

    rows = load(None) if os.path.isdir(PROJECTS) else []
    data = {"now": int(time.time()), "account": account, "five": five, "seven": seven,
            "spend": spend_data(rows)}
    # "</" would close the <script> block the data is embedded in.
    template = importlib.resources.files(__package__).joinpath("graph.html").read_text(encoding="utf-8")
    page = template.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"  wrote {path}  ({len(five)} five-hour and {len(seven)} weekly windows, "
          f"{len(rows)} requests from {len(data['spend']['projects'])} projects)")
    return 0


def field(payload: object, *keys: str) -> int | float | None:
    """The number at this path of a JSON object, or None if anything else is there."""
    for key in keys:
        payload = payload.get(key) if isinstance(payload, dict) else None
    return payload if isinstance(payload, (int, float)) and not isinstance(payload, bool) else None


def guard_setting(name: str) -> float:
    """A guard threshold. A bad value falls back on the default: the guard runs on
    every prompt, where an error message would reach nobody who can act on it."""
    try:
        value = float(os.environ[f"CLAUDE_USAGE_{name}"])
    except (KeyError, ValueError):
        return GUARD_DEFAULTS[name]
    return value if 0 <= value < float("inf") else GUARD_DEFAULTS[name]


def tail(path: str, size: int) -> str:
    """The last `size` bytes of a file, which is empty when it cannot be read.
    Transcripts grow to many megabytes, and only their end matters here."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - size))
            return fh.read().decode(errors="replace")
    except OSError:
        return ""


def last_activity(transcript: str) -> int | None:
    """Time of the last line of a transcript. No payload has a cache field, so the
    age of that line is the only measure of how long the prompt cache has left."""
    stamps = re.findall(r'"timestamp":"([^"]+)"', tail(transcript, 65536))
    try:
        return int(dt.datetime.fromisoformat(stamps[-1]).timestamp())
    except (IndexError, ValueError):
        return None


def context_file(transcript: str) -> str:
    """State file of a session. The transcript is named after the session, so two
    sessions never share one."""
    session = os.path.basename(transcript).removesuffix(".jsonl")
    return os.path.join(STATE_DIR, f"ctx-{session}.txt")


def write_context(payload: dict, transcript: str) -> None:
    """Publish the status line's figures for `guard`, which receives none of them:
    no hook input carries the context size, the cost or the quotas."""
    usage = (field(payload, "context_window", "current_usage", key)
             for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    used = field(payload, "context_window", "used_percentage")
    window = field(payload, "context_window", "context_window_size") or 0
    # The three input buckets are disjoint, so their sum is the real prompt size.
    tokens = sum(n or 0 for n in usage) or int(window * (used or 0) / 100)

    def whole(value: int | float | None) -> str:
        return "" if value is None else str(int(value))

    activity = last_activity(transcript)
    figures = {
        "pct": "" if used is None else f"{used:.0f}",
        "tokens": tokens,
        "window": window,
        "cost": field(payload, "cost", "total_cost_usd") or 0,
        "five": whole(field(payload, "rate_limits", "five_hour", "used_percentage")),
        "seven": whole(field(payload, "rate_limits", "seven_day", "used_percentage")),
        "five_reset": whole(field(payload, "rate_limits", "five_hour", "resets_at")),
        "seven_reset": whole(field(payload, "rate_limits", "seven_day", "resets_at")),
        "cache_expiry": int(activity + guard_setting("CACHE_TTL")) if activity else 0,
    }
    # Renamed into place, so `guard` never reads a half-written file.
    path = context_file(transcript)
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w") as fh:
        fh.writelines(f"{key}={value}\n" for key, value in figures.items())
    os.replace(temporary, path)


def append_quota(payload: dict, now: int) -> None:
    """Add the official percentages to the history when one of its values changed."""
    five = field(payload, "rate_limits", "five_hour", "used_percentage")
    seven = field(payload, "rate_limits", "seven_day", "used_percentage")
    five_reset = field(payload, "rate_limits", "five_hour", "resets_at")
    seven_reset = field(payload, "rate_limits", "seven_day", "resets_at")
    if five is None and seven is None:
        return

    # Each config directory is one account, with limits of its own.
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR") or ".claude"
    account = os.path.basename(config_dir.rstrip(os.sep))
    key = "|".join("" if v is None else str(v) for v in (five, seven, five_reset, seven_reset))
    last_file = os.path.join(STATE_DIR, f"quota-last-{account}.txt")
    try:
        with open(last_file) as fh:
            if fh.read() == key:
                return
    except OSError:
        pass
    with open(last_file, "w") as fh:
        fh.write(key)
    line = json.dumps({"ts": now, "account": account, "five": five, "seven": seven,
                       "five_reset": five_reset, "seven_reset": seven_reset},
                      separators=(",", ":"))
    with open(QUOTA_HISTORY, "a") as fh:
        fh.write(line + "\n")


def sweep_context_files(now: int) -> None:
    """Delete the state files of old sessions, at most once a day."""
    marker = os.path.join(STATE_DIR, ".last-sweep")
    try:
        if now - os.path.getmtime(marker) <= SWEEP_EVERY:
            return
    except OSError:
        pass
    for path in glob.glob(os.path.join(STATE_DIR, "ctx-*.txt")):
        try:
            if now - os.path.getmtime(path) > CONTEXT_FILE_MAX_AGE:
                os.remove(path)
        except OSError:
            continue  # another session swept it first
    with open(marker, "a"):
        os.utime(marker)


def record() -> int:
    """`claude-usage record`: save what Claude Code hands a status line on stdin.

    Prints nothing on stdout, which is what a status line displays.
    """
    try:
        payload = json.load(sys.stdin)
    except ValueError as error:
        print(f"claude-usage record: stdin is not JSON: {error}", file=sys.stderr)
        return 1
    if not isinstance(payload, dict):
        print("claude-usage record: stdin is not a JSON object.", file=sys.stderr)
        return 1

    now = int(time.time())
    transcript = payload.get("transcript_path")
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        if transcript and isinstance(transcript, str):
            write_context(payload, transcript)
        append_quota(payload, now)
        sweep_context_files(now)
    except OSError as error:
        print(f"claude-usage record: {error}", file=sys.stderr)
        return 1
    return 0


def read_context(transcript: str, now: int) -> dict[str, str]:
    """The figures `record` last published for this session, unless they are stale.
    Parsed key by key, so a mangled file only loses the keys it mangled."""
    path = context_file(transcript)
    try:
        if now - os.path.getmtime(path) > guard_setting("STATE_MAX_AGE"):
            return {}
        with open(path, errors="replace") as fh:
            return dict(line.rstrip("\n").partition("=")[::2] for line in fh)
    except OSError:
        return {}


def last_prompt_size(transcript: str) -> int:
    """Size of the prompt the next request resends, from the last assistant turn.
    A status line only redraws when Claude Code repaints, so an idle session has no
    fresh figures. Sidechain turns belong to a subagent's context, not this one's."""
    for line in reversed(tail(transcript, 1_000_000).splitlines()):
        if '"assistant"' not in line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # the first line of the slice is usually cut
        if isinstance(entry, dict) and entry.get("type") == "assistant" \
                and not entry.get("isSidechain"):
            return int(sum(field(entry, "message", "usage", key) or 0
                           for key in ("input_tokens", "cache_creation_input_tokens",
                                       "cache_read_input_tokens")))
    return 0


def pace_gap(pct: int, reset: int | None, span: int, now: int) -> tuple[int | None, int | None]:
    """The pace is the share of the window already elapsed, where usage would sit if
    the quota were spent evenly until the reset. Returns it with how far `pct` is
    ahead of it, or (None, None) when the reset time is unknown."""
    if not reset or reset <= 0:
        return None, None
    pace = min(100, max(0, (span - (reset - now)) * 100 // span))
    return pace, pct - pace


def quota_crossed(pct: int | None, gap: int | None, fallback: str) -> bool:
    """A raw 73% is fine three hours into a 5-hour window: what matters is the gap to
    the pace. Past QUOTA_HARD one large request can reach the limit, whatever the pace."""
    if pct is None or pct < 0:
        return False
    if pct >= guard_setting("QUOTA_HARD"):
        return True
    if gap is not None:
        return gap > guard_setting("PACE_GAP")
    return pct >= guard_setting(fallback)


def format_quota(label: str, pct: int | None, pace: int | None, gap: int | None) -> str:
    if pct is None:
        return ""
    if gap is None:
        return f" · {label} {pct}%"
    return f" · {label} {pct}% (pace {pace}%, {'±0' if gap == 0 else f'{gap:+d}'})"


def guard() -> int:
    """`claude-usage guard`: UserPromptSubmit hook printing a [budget] line for Claude.

    Claude cannot see how full its own context is, so a rule such as "warn me past
    150k tokens" needs this measure to be followed. UserPromptSubmit is the event
    whose stdout reaches Claude before the request is sent. Below every threshold
    nothing is printed, so most prompts cost no tokens.

    It reports which thresholds were crossed and never what to do about them: that
    belongs in the user's CLAUDE.md. It always lets the prompt through, because a
    hook that exits with 2 would reject it.
    """
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0
    transcript = payload.get("transcript_path") if isinstance(payload, dict) else None
    if not transcript or not isinstance(transcript, str):
        return 0

    def whole(key: str) -> int | None:
        try:
            return int(state.get(key, ""))
        except ValueError:
            return None

    now = int(time.time())
    state = read_context(transcript, now)
    tokens = whole("tokens") or last_prompt_size(transcript)
    if tokens <= 0:
        return 0

    window = whole("window")
    window = window if window and window > 0 else 1_000_000
    pct = whole("pct")
    pct = tokens * 100 // window if pct is None else pct
    try:
        cost = float(state.get("cost", 0))
    except ValueError:
        cost = 0.0
    five, seven = whole("five"), whole("seven")
    cache_expiry = whole("cache_expiry")
    if not cache_expiry:
        activity = last_activity(transcript)
        cache_expiry = int(activity + guard_setting("CACHE_TTL")) if activity else 0

    five_pace, five_gap = (None, None) if five is None else \
        pace_gap(five, whole("five_reset"), QUOTA_SPAN["five"], now)
    seven_pace, seven_gap = (None, None) if seven is None else \
        pace_gap(seven, whole("seven_reset"), QUOTA_SPAN["seven"], now)
    cold = bool(cache_expiry) and now >= cache_expiry and tokens >= guard_setting("CACHE_MIN")

    crossed = []
    if tokens >= guard_setting("CONTEXT_STOP"):
        crossed.append("CONTEXT_STOP")
    elif tokens >= guard_setting("CONTEXT_DECISION"):
        crossed.append("CONTEXT_DECISION")
    if cost >= guard_setting("COST_WARN"):
        crossed.append("COST")
    if quota_crossed(five, five_gap, "FIVE_WARN"):
        crossed.append("QUOTA_5H")
    if quota_crossed(seven, seven_gap, "SEVEN_WARN"):
        crossed.append("QUOTA_7D")
    if cold:
        crossed.append("COLD_CACHE")
    if not crossed:
        return 0

    line = f"context {tokens // 1000}k ({pct}%)"
    if cost > 0:
        line += f" · ${cost:.2f}"
    line += format_quota("5h", five, five_pace, five_gap)
    line += format_quota("7d", seven, seven_pace, seven_gap)
    if cold:
        line += f" · cold cache since {dt.datetime.fromtimestamp(cache_expiry):%H:%M}"
    print(f"[budget] {line}")
    print(f"[budget] thresholds crossed: {','.join(crossed)}")
    return 0


COMMANDS = {"record": record, "guard": guard}


def positive_budget(value: str) -> float:
    budget = float(value)
    if not 0 < budget < float("inf"):
        raise ValueError(value)
    return budget


def main() -> int:
    # Ahead of the parser, which exits with 2 on a bad CLAUDE_USAGE_*_BUDGET value:
    # neither command reads those, and a hook exiting with 2 rejects the user's prompt.
    if sys.argv[1:] in (["record"], ["guard"]):
        return COMMANDS[sys.argv[1]]()

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", title="commands to run from Claude Code")
    commands.add_parser("record", help="status line: save the figures Claude Code sends on stdin")
    commands.add_parser("guard", help="UserPromptSubmit hook: warn when a threshold is crossed")
    parser.add_argument("--days", type=int, default=28, help="window to analyse (default 28)")
    parser.add_argument("--sessions", type=int, default=12, help="rows in the session table")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    parser.add_argument("--csv", metavar="PATH", help="also dump every request to a CSV")
    parser.add_argument("--graph", nargs="?", const=QUOTA_GRAPH, metavar="PATH",
                        help="write an HTML page of the recorded 5-hour and weekly quota and "
                             "of the spend, both cut by its time range filter over every "
                             f"transcript, --days not applying (default {QUOTA_GRAPH}) and exit")
    parser.add_argument("--web", type=float, metavar="PERCENT",
                        help="record the claude.ai share of the week in progress, as shown by "
                             "claude.ai's usage page, and exit")
    # argparse converts a string default with `type`, so a bad environment value
    # is reported like a bad flag.
    parser.add_argument("--session-budget", type=positive_budget, metavar="DOLLARS",
                        default=os.environ.get("CLAUDE_USAGE_SESSION_BUDGET"),
                        help="$ equivalent of a full 5-hour window, instead of the measured one "
                             "(env: CLAUDE_USAGE_SESSION_BUDGET)")
    parser.add_argument("--weekly-budget", type=positive_budget, metavar="DOLLARS",
                        default=os.environ.get("CLAUDE_USAGE_WEEKLY_BUDGET"),
                        help="$ equivalent of a full week, instead of the measured one "
                             "(env: CLAUDE_USAGE_WEEKLY_BUDGET)")
    args = parser.parse_args()

    if args.command:
        parser.error(f"{args.command} takes no option")

    if args.web is not None:
        return record_web(args.web)

    if args.graph:
        return write_graph(args.graph)

    if not os.path.isdir(PROJECTS):
        print(f"{PROJECTS} not found — is Claude Code installed?", file=sys.stderr)
        return 1

    rows = load(args.days)
    budget = budgets(rows, args.days, args.session_budget, args.weekly_budget)

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    if args.json:
        json.dump({
            "days": args.days,
            "total": sum(r["cost"] for r in rows),
            "budgets": budget,
            "weekly": [{"week": str(w), "cost": c} for w, c in weekly(rows)],
            "daily": [{"day": str(d), "peak_5h": p, "total": t} for d, p, t in daily_peaks(rows)],
            "sessions": [{**s, "start": s["start"].isoformat()} for s in sessions(rows)],
        }, sys.stdout, indent=2)
        return 0

    report(rows, args.days, args.sessions, budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())
