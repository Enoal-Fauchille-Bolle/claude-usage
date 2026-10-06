#!/usr/bin/env python3
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
two constants below stand in when that history is too thin.

Usage:
    claude-usage                 # last 28 days
    claude-usage --days 7        # shorter window
    claude-usage --sessions 20   # show more sessions in the session table
    claude-usage --json          # machine-readable output
    claude-usage --graph         # HTML page: official 5-hour/weekly quota, spend by project/model
    claude-usage --graph out.html
    claude-usage --web 22        # claude.ai took 22% of this week's usage
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import datetime as dt
import glob
import json
import os
import statistics
import sys
import time

# --- Calibration -----------------------------------------------------------
# The budgets are measured from the quota history when it allows (see
# budgets()); these constants are the fallback. Measured on 2026-10-06 on a Pro
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

# The status line appends one reading of the official /usage percentages here
# each time they change (see claude/.claude/statusline-command.sh).
QUOTA_HISTORY = os.path.expanduser("~/.claude/state/quota-history.jsonl")
QUOTA_GRAPH = os.path.expanduser("~/.claude/state/quota-graph.html")
QUOTA_SPAN = {"five": 5 * 3600, "seven": 7 * 86400}  # window length, seconds
# claude-usage --web appends the claude.ai share of each week here, as read on
# the "usage this week by product" panel of claude.ai.
WEB_SHARE = os.path.expanduser("~/.claude/state/web-share.jsonl")

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

        # The default dir keeps its account file in $HOME, alternates keep it inside.
        default = os.path.expanduser("~/.claude")
        account_file = (os.path.expanduser("~/.claude.json") if config_dir == default
                        else os.path.join(config_dir, ".claude.json"))
        label = os.path.basename(config_dir)
        try:
            with open(account_file) as fh:
                label = json.load(fh).get("oauthAccount", {}).get("emailAddress", label)
        except (OSError, json.JSONDecodeError):
            pass

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


def load(days: int) -> list[dict]:
    """Replay every assistant request from the transcripts into a flat list.

    Deduplicated on requestId: a single assistant turn is written to the
    transcript once per content block, so counting lines would double-count.
    Requests from a Claude model missing in PRICES are left out and reported
    on stderr, so a new model shows up as a warning rather than as a quiet dip.
    """
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
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
                if when < cutoff:
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


def sessions(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        grouped[r["session"]].append(r)

    out = []
    for sid, rs in grouped.items():
        stale = 0.0
        gaps = 0
        for before, after in zip(rs, rs[1:]):
            idle = (after["t"] - before["t"]).total_seconds() / 3600
            if idle > CACHE_LIFETIME_H and after["write"] > 50_000:
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


def spend_data(rows: list[dict], days: int) -> dict:
    """Aggregate the requests for the spend half of the HTML page."""
    projects: dict[str, dict] = {}
    for s in sessions(rows):
        p = projects.setdefault(s["project"], {
            "name": s["project"], "path": s["path"], "cost": 0.0, "stale": 0.0,
            "long": 0.0, "sessions": 0, "requests": 0})
        p["cost"] += s["cost"]
        p["stale"] += s["stale_cost"]
        p["long"] += s["long_cost"]
        p["sessions"] += 1
        p["requests"] += s["requests"]
    ranked = sorted(projects.values(), key=lambda p: -p["cost"])

    # Colours follow the project, ranked over the whole window, so a day never
    # repaints them. Everything past the palette folds into the last slot.
    stacked = [p["name"] for p in ranked[:STACK_PROJECTS]]
    slot = {name: i for i, name in enumerate(stacked)}
    today = dt.date.today()
    daily = {today - dt.timedelta(days=i): [0.0] * (len(stacked) + 1) for i in range(days)}
    models: dict[str, dict] = {}
    for r in rows:
        day = r["t"].date()
        if day in daily:
            daily[day][slot.get(r["project"], len(stacked))] += r["cost"]
        m = models.setdefault(r["model"], {"name": model_label(r["model"]), "cost": 0.0, "requests": 0})
        m["cost"] += r["cost"]
        m["requests"] += 1

    return {
        "days": days,
        "total": sum(r["cost"] for r in rows),
        "projects": ranked,
        "stack": stacked + ["Other"],
        "daily": [{"day": str(d), "v": [round(c, 4) for c in v]} for d, v in sorted(daily.items())],
        "models": sorted(models.values(), key=lambda m: -m["cost"]),
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
        print(f"{QUOTA_HISTORY} not found — the status line records it while Claude Code runs.",
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


def budgets(rows: list[dict], days: int) -> dict:
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


GRAPH_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Claude usage</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --hover: rgba(11,11,11,0.05);
  --s1: #2a78d6; --s2: #eb6834;
  --c0: #2a78d6; --c1: #eb6834; --c2: #1baf7a; --c3: #eda100; --c4: #e87ba4; --c5: #008300;
  --c6: #4a3aa7; --c7: #b5b4ad;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --hover: rgba(255,255,255,0.08);
    --s1: #3987e5; --s2: #d95926;
    --c0: #3987e5; --c1: #d95926; --c2: #199e70; --c3: #c98500; --c4: #d55181; --c5: #008300;
    --c6: #9085e9; --c7: #5c5b57;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --hover: rgba(255,255,255,0.08);
  --s1: #3987e5; --s2: #d95926;
  --c0: #3987e5; --c1: #d95926; --c2: #199e70; --c3: #c98500; --c4: #d55181; --c5: #008300;
  --c6: #9085e9; --c7: #5c5b57;
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px; background: var(--page); color: var(--ink);
  font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1000px; margin: 0 auto; }
h1 { font-size: 20px; font-weight: 600; margin: 0 0 2px; }
.sub { color: var(--ink2); margin: 0 0 16px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
.filters { display: flex; gap: 4px; margin-bottom: 12px; }
.filters button { font: inherit; color: var(--ink2); background: none; border: 1px solid transparent;
  border-radius: 8px; padding: 4px 12px; cursor: pointer; }
.filters button:hover { background: var(--hover); }
.filters button[aria-pressed="true"] { color: var(--ink); font-weight: 600; border-color: var(--border); }
.legend { display: flex; flex-wrap: wrap; gap: 4px 24px; margin: 0 0 4px; }
.key { display: inline-flex; align-items: center; gap: 8px; color: var(--ink2); }
.key i { width: 16px; height: 0; border-top: 2px solid var(--c); border-radius: 1px; }
.key strong { color: var(--ink); font-weight: 600; }
.plot { position: relative; margin-bottom: 8px; }
.plot svg { display: block; overflow: visible; }
.plot svg:focus-visible { outline: 2px solid var(--muted); outline-offset: 4px; border-radius: 4px; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 12px; }
.tick.day { fill: var(--ink2); }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.s1 { --c: var(--s1); }
.s2 { --c: var(--s2); }
.line { stroke: var(--c); }
.dot { fill: var(--c); stroke: var(--surface); stroke-width: 2; }
.cross { stroke: var(--muted); stroke-width: 1; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.12); white-space: nowrap; z-index: 1; }
.tip .when { color: var(--ink2); margin-bottom: 4px; }
.tip .row { display: flex; align-items: center; gap: 8px; }
.tip .row i { width: 12px; border-top: 2px solid var(--c); }
.tip .row strong { min-width: 3ch; }
.tip .row span { color: var(--ink2); }
.note { color: var(--muted); margin: 8px 0 0; font-size: 12px; }
details { margin-top: 12px; }
summary { cursor: pointer; color: var(--ink2); }
table { border-collapse: collapse; margin: 8px 0 12px; }
th, td { text-align: left; padding: 2px 16px 2px 0; font-variant-numeric: tabular-nums; }
th { color: var(--ink2); font-weight: 500; border-bottom: 1px solid var(--grid); }
caption { text-align: left; color: var(--ink2); padding: 8px 0 4px; }
.card + .card { margin-top: 16px; }
h2 { font-size: 16px; font-weight: 600; margin: 0 0 2px; }
h3 { font-size: 14px; font-weight: 600; margin: 20px 0 4px; }
.duo { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 0 24px; }
.duo .note { margin: 0 0 8px; }
.bar { fill: var(--c); }
.k0 { --c: var(--c0); } .k1 { --c: var(--c1); } .k2 { --c: var(--c2); } .k3 { --c: var(--c3); }
.k4 { --c: var(--c4); } .k5 { --c: var(--c5); } .k6 { --c: var(--c6); } .k7 { --c: var(--c7); }
.lbl { fill: var(--ink); font-size: 12px; }
.val { fill: var(--ink2); font-size: 12px; font-variant-numeric: tabular-nums; }
.hl { fill: var(--hover); }
.key i.sw, .tip .row i.sw { width: 10px; height: 10px; border: 0; border-radius: 2px; background: var(--c); }
td.num, th.num { text-align: right; }
@media (forced-colors: active) { .line.s2 { stroke-dasharray: 6 4; } }
</style>
</head>
<body>
<main>
  <h1>Claude usage</h1>
  <p class="sub" id="sub"></p>
  <div class="card">
    <div class="filters" id="filters" role="group" aria-label="Time range"></div>
    <div id="charts"></div>
    <p class="note">Each line is one limit window; a break means the window ended or nothing was recorded.
      Readings below an earlier one in the same window are stale and ignored.</p>
    <details><summary>Table view</summary><div id="tables"></div></details>
  </div>
  <div class="card" id="spend" hidden>
    <h2>Claude Code spend</h2>
    <p class="sub" id="spend-sub"></p>
    <h3>By project</h3>
    <div class="plot" id="by-project"></div>
    <h3>Per day, by project</h3>
    <div id="per-day"></div>
    <h3>By model</h3>
    <div class="plot" id="by-model"></div>
    <div class="duo">
      <div>
        <h3>Cold-cache rewrites</h3>
        <p class="note">Re-writing the cache after more than an hour idle.</p>
        <div class="plot" id="waste-stale"></div>
      </div>
      <div>
        <h3>Context over 150k</h3>
        <p class="note">Everything spent on requests carrying more than 150k tokens.</p>
        <div class="plot" id="waste-long"></div>
      </div>
    </div>
    <p class="note">Both overlap: a cold rewrite of a long context counts in each.</p>
    <details><summary>Table view</summary><div id="spend-tables"></div></details>
  </div>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
(() => {
const D = JSON.parse(document.getElementById("data").textContent);
const NS = "http://www.w3.org/2000/svg";
const HOUR = 3600, DAY = 86400;
const SERIES = [
  { key: "five", label: "5-hour session", css: "s1", segs: D.five },
  { key: "seven", label: "Weekly", css: "s2", segs: D.seven },
];
const times = SERIES.flatMap(s => s.segs.flatMap(g => g.points.map(p => p[0])));
const first = Math.min(...times), span = D.now - first;
const presets = [["24 h", DAY], ["3 days", 3 * DAY], ["7 days", 7 * DAY]].filter(([, r]) => r < span);
presets.push(["All", 0]);

// A weekly window is 7 days long, so that is the view that shows a whole one.
let sel = Math.max(presets.findIndex(([name]) => name === "7 days"), 0) || presets.length - 1;

const fmtWhen = new Intl.DateTimeFormat(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
const fmtDay = new Intl.DateTimeFormat(undefined, { weekday: "short", day: "numeric" });
const fmtHour = new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" });

const svg = (tag, attrs, parent) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
};
const html = (tag, cls, text, parent) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
};

const domain = () => [presets[sel][1] ? Math.max(D.now - presets[sel][1], first) : first, D.now];

// Value of a series at time t: the last reading of the window that covers t.
function valueAt(s, t) {
  for (const g of s.segs) {
    if (t >= g.points[0][0] && t <= g.end) {
      let v = g.points[0][1];
      for (const p of g.points) if (p[0] <= t) v = p[1];
      return v;
    }
  }
  return null;
}

// The current reading, or null when the latest window has already ended.
function current(s) {
  const g = s.segs[s.segs.length - 1];
  return g && g.reset > D.now ? g.points[g.points.length - 1][1] : null;
}

function ticks(x0, x1) {
  const steps = [HOUR, 2 * HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, DAY, 2 * DAY, 7 * DAY];
  const step = steps.find(s => (x1 - x0) / s <= 8) ?? 7 * DAY;
  const d = new Date(x0 * 1000);
  if (step >= DAY) d.setHours(0, 0, 0, 0);
  else { d.setMinutes(0, 0, 0); d.setHours(Math.floor(d.getHours() / (step / HOUR)) * (step / HOUR)); }
  const out = [];
  for (let t = d.getTime() / 1000; t <= x1; t += step) if (t >= x0) out.push(t);
  return out;
}

function legendKey(s, parent) {
  const k = html("span", "key", undefined, parent);
  html("i", s.css, undefined, k);
  html("span", "", s.label, k);
  const v = current(s);
  html("strong", "", v === null ? "no active window" : v + "%", k);
}

let uid = 0;
function drawChart(host, series, height) {
  const width = Math.max(320, host.clientWidth);
  const m = { l: 44, r: 16, t: 12, b: 28 };
  const [x0, x1] = domain();
  const X = t => m.l + (t - x0) / (x1 - x0) * (width - m.l - m.r);
  const Y = v => height - m.b - v / 100 * (height - m.t - m.b);
  const id = "clip" + uid++;

  const root = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", tabindex: 0,
    "aria-label": series.map(s => s.label).join(" and ") + " usage over time; the table view below has the values" }, host);
  const clip = svg("clipPath", { id }, svg("defs", {}, root));
  svg("rect", { x: m.l, y: m.t, width: width - m.l - m.r, height: height - m.t - m.b }, clip);

  for (const v of [0, 25, 50, 75, 100]) {
    svg("line", { class: v === 0 ? "axis" : "grid", x1: m.l, x2: width - m.r, y1: Y(v), y2: Y(v) }, root);
    svg("text", { class: "tick", x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, root).textContent = v + "%";
  }
  for (const t of ticks(x0, x1)) {
    const d = new Date(t * 1000);
    const midnight = d.getHours() === 0 && d.getMinutes() === 0;
    svg("line", { class: "axis", x1: X(t), x2: X(t), y1: Y(0), y2: Y(0) + 4 }, root);
    svg("text", { class: "tick" + (midnight ? " day" : ""), x: X(t), y: height - 8, "text-anchor": "middle" }, root)
      .textContent = midnight ? fmtDay.format(d) : fmtHour.format(d);
  }

  const plot = svg("g", { "clip-path": `url(#${id})` }, root);
  for (const s of series) {
    for (const g of s.segs) {
      if (g.end < x0 || g.points[0][0] > x1) continue;
      let d = `M${X(g.points[0][0])} ${Y(g.points[0][1])}`;
      for (const p of g.points.slice(1)) d += `H${X(p[0])}V${Y(p[1])}`;
      d += `H${X(g.end)}`;
      svg("path", { class: "line " + s.css, d }, plot);
    }
    const last = s.segs[s.segs.length - 1];
    if (last && last.end >= x0) {
      svg("circle", { class: "dot " + s.css, cx: X(last.end), cy: Y(last.points[last.points.length - 1][1]), r: 4 }, root);
    }
  }

  // Hover layer: a crosshair snapped to the nearest reading, one tooltip for every series.
  const snaps = [...new Set(series.flatMap(s => s.segs.flatMap(g => g.points.map(p => p[0]).concat(g.end))))]
    .filter(t => t >= x0 && t <= x1).sort((a, b) => a - b);
  const cross = svg("line", { class: "cross", y1: m.t, y2: Y(0), visibility: "hidden" }, root);
  const dots = series.map(s => svg("circle", { class: "dot " + s.css, r: 4, visibility: "hidden" }, root));
  const tip = html("div", "tip", undefined, host);
  tip.hidden = true;

  const hide = () => { cross.setAttribute("visibility", "hidden"); dots.forEach(e => e.setAttribute("visibility", "hidden")); tip.hidden = true; };
  const show = i => {
    if (!snaps.length) return;
    const t = snaps[i];
    cross.setAttribute("x1", X(t)); cross.setAttribute("x2", X(t)); cross.setAttribute("visibility", "visible");
    tip.replaceChildren(html("div", "when", fmtWhen.format(new Date(t * 1000))));
    const readings = [];
    series.forEach((s, j) => {
      const v = valueAt(s, t);
      if (v !== null) readings.push(v);
      const row = html("div", "row", undefined, tip);
      html("i", s.css, undefined, row);
      html("strong", "", v === null ? "–" : v + "%", row);
      html("span", "", s.label, row);
      if (v === null) dots[j].setAttribute("visibility", "hidden");
      else { dots[j].setAttribute("cx", X(t)); dots[j].setAttribute("cy", Y(v)); dots[j].setAttribute("visibility", "visible"); }
    });
    tip.hidden = false;
    tip.style.top = (Math.max(...readings, 0) > 50 ? Y(0) - tip.offsetHeight - 8 : m.t + 4) + "px";
    const flip = X(t) > width * 0.6;
    tip.style.left = flip ? "auto" : X(t) + 12 + "px";
    tip.style.right = flip ? width - X(t) + 12 + "px" : "auto";
  };

  let cur = snaps.length - 1;
  const hit = svg("rect", { x: m.l, y: m.t, width: width - m.l - m.r, height: height - m.t - m.b, fill: "transparent" }, root);
  hit.addEventListener("pointermove", e => {
    const px = e.clientX - root.getBoundingClientRect().left;
    const t = x0 + (px - m.l) / (width - m.l - m.r) * (x1 - x0);
    cur = snaps.reduce((best, s, i) => Math.abs(s - t) < Math.abs(snaps[best] - t) ? i : best, 0);
    show(cur);
  });
  hit.addEventListener("pointerleave", hide);
  root.addEventListener("focus", () => show(cur));
  root.addEventListener("blur", hide);
  root.addEventListener("keydown", e => {
    if (e.key === "ArrowLeft") cur = Math.max(0, cur - 1);
    else if (e.key === "ArrowRight") cur = Math.min(snaps.length - 1, cur + 1);
    else if (e.key === "Escape") return hide();
    else return;
    e.preventDefault(); show(cur);
  });
}

function render() {
  const charts = document.getElementById("charts");
  charts.replaceChildren();
  const legend = html("div", "legend", undefined, charts);
  SERIES.forEach(s => legendKey(s, legend));
  drawChart(html("div", "plot", undefined, charts), SERIES, 320);
  const filters = document.getElementById("filters");
  filters.replaceChildren();
  filters.hidden = presets.length < 2;
  presets.forEach(([name], i) => {
    const b = html("button", "", name, filters);
    b.type = "button";
    b.setAttribute("aria-pressed", String(i === sel));
    b.addEventListener("click", () => { sel = i; render(); });
  });
}

function tables() {
  const host = document.getElementById("tables");
  const fmt = t => fmtWhen.format(new Date(t * 1000));
  for (const s of SERIES) {
    const table = html("table", "", undefined, host);
    html("caption", "", s.label, table);
    const head = html("tr", "", undefined, html("thead", "", undefined, table));
    ["First reading", "Last reading", "Window resets", "Peak"].forEach(h => html("th", "", h, head));
    const body = html("tbody", "", undefined, table);
    for (const g of s.segs) {
      const row = html("tr", "", undefined, body);
      [fmt(g.points[0][0]), fmt(g.end), fmt(g.reset), g.points[g.points.length - 1][1] + "%"].forEach(c => html("td", "", c, row));
    }
  }
}

// --- Spend: dollars estimated from the transcripts, a separate unit from the quota %.
const S = D.spend;
const fmtUsd = v => "$" + (v >= 100 ? Math.round(v).toLocaleString() : v.toFixed(v >= 10 ? 1 : 2));
const fmtPct = v => (S.total ? Math.round(v / S.total * 100) : 0) + "%";
const fmtDate = new Intl.DateTimeFormat(undefined, { weekday: "short", day: "numeric", month: "short" });
const dayOf = s => new Date(s + "T00:00:00");
const slot = j => "k" + Math.min(j, 7);

// A clean axis: four or five steps of 1, 2 or 5 times a power of ten.
function scale(max) {
  if (!(max > 0)) return { top: 1, step: 1 };
  const raw = max / 4, p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map(k => k * p).find(s => s >= raw);
  return { top: Math.ceil(max / step) * step, step };
}

// Bars round their data end only; the baseline end stays square.
const barRight = (x, y, w, h) => {
  const r = Math.min(4, w, h / 2);
  return `M${x} ${y}H${x + w - r}Q${x + w} ${y} ${x + w} ${y + r}V${y + h - r}Q${x + w} ${y + h} ${x + w - r} ${y + h}H${x}Z`;
};
const barUp = (x, y, w, h) => {
  const r = Math.min(4, h, w / 2);
  return `M${x} ${y + h}V${y + r}Q${x} ${y} ${x + r} ${y}H${x + w - r}Q${x + w} ${y} ${x + w} ${y + r}V${y + h}Z`;
};

// Shorten an SVG label with an ellipsis until it fits; the tooltip keeps the full text.
function fit(t, max) {
  const full = t.textContent;
  for (let n = full.length; n > 1 && t.getComputedTextLength() > max; n--) t.textContent = full.slice(0, n - 1) + "…";
}

function tipRows(tip, title, rows) {
  tip.replaceChildren(html("div", "when", title));
  for (const [css, value, label] of rows) {
    const row = html("div", "row", undefined, tip);
    if (css) html("i", "sw " + css, undefined, row);
    html("strong", "", value, row);
    html("span", "", label, row);
  }
}

// Hover and keyboard layer shared by the bar charts: one band per mark, bigger than
// the mark, a wash behind the hovered band and one tooltip.
function interact(root, host, width, height, n, band, show) {
  const hl = svg("rect", { class: "hl", rx: 6, visibility: "hidden" });
  root.insertBefore(hl, root.firstChild);
  const tip = html("div", "tip", undefined, host);
  tip.hidden = true;
  let cur = -1;
  const on = i => {
    cur = i;
    const [x, y, w, h] = band(i);
    Object.entries({ x, y, width: w, height: h }).forEach(([k, v]) => hl.setAttribute(k, v));
    hl.setAttribute("visibility", "visible");
    const [ax, ay] = show(i, tip);
    tip.hidden = false;
    const flip = ax > width * 0.6;
    tip.style.left = flip ? "auto" : ax + 12 + "px";
    tip.style.right = flip ? width - ax + 12 + "px" : "auto";
    tip.style.top = Math.max(0, Math.min(ay - tip.offsetHeight / 2, height - tip.offsetHeight)) + "px";
  };
  const off = () => { hl.setAttribute("visibility", "hidden"); tip.hidden = true; };
  for (let i = 0; i < n; i++) {
    const [x, y, w, h] = band(i);
    svg("rect", { x, y, width: w, height: h, fill: "transparent" }, root).addEventListener("pointerenter", () => on(i));
  }
  root.addEventListener("pointerleave", off);
  root.addEventListener("focus", () => on(Math.max(cur, 0)));
  root.addEventListener("blur", off);
  root.addEventListener("keydown", e => {
    if (e.key === "Escape") return off();
    const step = { ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1 }[e.key];
    if (!step) return;
    e.preventDefault();
    on(Math.min(n - 1, Math.max(0, cur + step)));
  });
}

// Horizontal bars, largest first, each labelled with its value at the tip.
// items: [{ label, value, title, rows }]; `max` shares one scale across charts.
function hbar(host, items, aria, max) {
  host.replaceChildren();
  if (!items.length) { html("p", "note", "Nothing in this window.", host); return; }
  const width = Math.max(280, host.clientWidth);
  const row = 28, thick = 16, labelW = Math.min(150, Math.round(width * 0.34)), valW = 64;
  const height = items.length * row;
  const x0 = labelW + 8, plotW = width - x0 - valW;
  max = max ?? Math.max(...items.map(d => d.value));
  const W = v => max > 0 ? v / max * plotW : 0;
  const root = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", tabindex: 0,
    "aria-label": aria + "; the table view below has the values" }, host);
  items.forEach((d, i) => {
    const cy = i * row + row / 2;
    const t = svg("text", { class: "lbl", x: labelW, y: cy + 4, "text-anchor": "end" }, root);
    t.textContent = d.label;
    fit(t, labelW);
    if (W(d.value) >= 0.5) svg("path", { class: "bar k0", d: barRight(x0, cy - thick / 2, W(d.value), thick) }, root);
    svg("text", { class: "val", x: x0 + W(d.value) + 6, y: cy + 4 }, root).textContent = fmtUsd(d.value);
  });
  svg("line", { class: "axis", x1: x0, x2: x0, y1: 0, y2: height }, root);
  interact(root, host, width, height, items.length, i => [0, i * row, width, row],
    (i, tip) => { tipRows(tip, items[i].title, items[i].rows); return [x0 + W(items[i].value), i * row + row / 2]; });
}

// One column per day, stacked by project, a 2px surface gap between segments.
function perDay(host) {
  host.replaceChildren();
  const legend = html("div", "legend", undefined, host);
  S.stack.forEach((name, j) => {
    if (!S.daily.some(d => d.v[j] > 0)) return;
    const k = html("span", "key", undefined, legend);
    html("i", "sw " + slot(j), undefined, k);
    html("span", "", name, k);
  });
  const plot = html("div", "plot", undefined, host);
  const width = Math.max(320, plot.clientWidth), height = 260;
  const m = { l: 52, r: 8, t: 12, b: 28 };
  const n = S.daily.length, bandW = (width - m.l - m.r) / n, bw = Math.max(2, Math.min(24, bandW - 4));
  const totals = S.daily.map(d => d.v.reduce((a, b) => a + b, 0));
  const { top, step } = scale(Math.max(...totals));
  const Y = v => height - m.b - v / top * (height - m.t - m.b);
  const root = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", tabindex: 0,
    "aria-label": "Spend per day, stacked by project; the table view below has the values" }, plot);

  for (let v = 0; v <= top + step / 2; v += step) {
    svg("line", { class: v === 0 ? "axis" : "grid", x1: m.l, x2: width - m.r, y1: Y(v), y2: Y(v) }, root);
    svg("text", { class: "tick", x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, root).textContent = "$" + v.toLocaleString();
  }
  // At most eight day labels, counted back from today so the last column is always named.
  const every = Math.ceil(n / 8);
  S.daily.forEach((d, i) => {
    const cx = m.l + bandW * (i + 0.5);
    if ((n - 1 - i) % every === 0) {
      svg("text", { class: "tick", x: cx, y: height - 8, "text-anchor": "middle" }, root).textContent = fmtDay.format(dayOf(d.day));
    }
    const segs = d.v.map((v, j) => [v, j]).filter(([v]) => v > 0);
    let y = Y(0);
    segs.forEach(([v, j], s) => {
      const h = Y(0) - Y(v), gap = s ? 2 : 0, yTop = y - h;
      y = yTop;
      if (h - gap < 0.5) return;
      const d = s === segs.length - 1 ? barUp(cx - bw / 2, yTop, bw, h - gap) : `M${cx - bw / 2} ${yTop}h${bw}v${h - gap}h${-bw}Z`;
      svg("path", { class: "bar " + slot(j), d }, root);
    });
  });

  interact(root, plot, width, height, n, i => [m.l + bandW * i, m.t, bandW, Y(0) - m.t], (i, tip) => {
    const d = S.daily[i];
    const rows = d.v.map((v, j) => [slot(j), fmtUsd(v), S.stack[j], v]).filter(r => r[3] > 0).sort((a, b) => b[3] - a[3]);
    tipRows(tip, fmtDate.format(dayOf(d.day)) + " · " + fmtUsd(totals[i]), rows.map(r => r.slice(0, 3)));
    return [m.l + bandW * (i + 0.5), height / 2];
  });
}

// Every project past the first ten folds into one row, so the chart stays readable.
function projectItems() {
  const top = S.projects.slice(0, 10), rest = S.projects.slice(10);
  const items = top.map(p => ({
    label: p.name, value: p.cost, title: p.path,
    rows: [["", fmtUsd(p.cost), fmtPct(p.cost) + " of the total"], ["", String(p.sessions), p.sessions === 1 ? "session" : "sessions"],
      ["", fmtUsd(p.stale), "cold-cache rewrites"], ["", fmtUsd(p.long), "over 150k context"]],
  }));
  if (rest.length) {
    const cost = rest.reduce((a, p) => a + p.cost, 0);
    items.push({ label: `${rest.length} others`, value: cost, title: rest.map(p => p.name).join(", "),
      rows: [["", fmtUsd(cost), fmtPct(cost) + " of the total"]] });
  }
  return items;
}

function renderSpend() {
  if (!S || !S.total) return;
  hbar(document.getElementById("by-project"), projectItems(), "Spend by project");
  perDay(document.getElementById("per-day"));
  hbar(document.getElementById("by-model"), S.models.map(md => ({
    label: md.name, value: md.cost, title: md.name,
    rows: [["", fmtUsd(md.cost), fmtPct(md.cost) + " of the total"], ["", md.requests.toLocaleString(), "requests"]],
  })), "Spend by model");

  // Same projects, same order and one shared scale, so the two sides compare.
  const waste = S.projects.slice(0, 8);
  const max = Math.max(...waste.flatMap(p => [p.stale, p.long]));
  for (const [id, key, what] of [["waste-stale", "stale", "cold-cache rewrites"], ["waste-long", "long", "over 150k context"]]) {
    hbar(document.getElementById(id), waste.map(p => ({
      label: p.name, value: p[key], title: p.path,
      rows: [["", fmtUsd(p[key]), what], ["", p.cost ? Math.round(p[key] / p.cost * 100) + "%" : "–", "of the project's spend"]],
    })), "Spend on " + what + " by project", max);
  }
}

function spendTables() {
  const host = document.getElementById("spend-tables");
  const table = (caption, head, rows) => {
    const t = html("table", "", undefined, host);
    html("caption", "", caption, t);
    const tr = html("tr", "", undefined, html("thead", "", undefined, t));
    head.forEach((h, i) => html("th", i ? "num" : "", h, tr));
    const body = html("tbody", "", undefined, t);
    for (const r of rows) {
      const row = html("tr", "", undefined, body);
      r.forEach((c, i) => html("td", i ? "num" : "", c, row));
    }
  };
  table("By project", ["Project", "Spend", "Share", "Sessions", "Cold cache", "Over 150k"],
    S.projects.map(p => [p.path, fmtUsd(p.cost), fmtPct(p.cost), p.sessions, fmtUsd(p.stale), fmtUsd(p.long)]));
  table("By model", ["Model", "Spend", "Share", "Requests"],
    S.models.map(md => [md.name, fmtUsd(md.cost), fmtPct(md.cost), md.requests.toLocaleString()]));
  table("Per day", ["Day", "Total", ...S.stack],
    S.daily.map(d => [fmtDate.format(dayOf(d.day)), fmtUsd(d.v.reduce((a, b) => a + b, 0)), ...d.v.map(fmtUsd)]));
}

document.getElementById("sub").textContent =
  "Official /usage percentages recorded by the status line" + (D.account ? " · account " + D.account : "");
tables();
render();
if (S && S.total) {
  document.getElementById("spend").hidden = false;
  document.getElementById("spend-sub").textContent = `${fmtUsd(S.total)} over the last ${S.days} days, ` +
    "estimated from the local transcripts at API list prices, every account on this machine. " +
    "A different unit from the quota above: claude.ai use is not in it.";
  spendTables();
  renderSpend();
}
let drawnWidth = document.getElementById("charts").clientWidth;
new ResizeObserver(() => {
  const w = document.getElementById("charts").clientWidth;
  if (w !== drawnWidth) { drawnWidth = w; render(); renderSpend(); }
}).observe(document.getElementById("charts"));
})();
</script>
</body>
</html>
"""


def write_graph(path: str, days: int) -> int:
    """Render the quota history and the Claude Code spend as a self-contained HTML
    page (no external assets)."""
    if not os.path.isfile(QUOTA_HISTORY):
        print(f"{QUOTA_HISTORY} not found — the status line records it while Claude Code runs.",
              file=sys.stderr)
        return 1

    account, records = load_quota()
    five = quota_segments(records, "five")
    seven = quota_segments(records, "seven")
    if not (five or seven):
        print("No usable quota readings recorded yet.", file=sys.stderr)
        return 1

    rows = load(days) if os.path.isdir(PROJECTS) else []
    data = {"now": int(time.time()), "account": account, "five": five, "seven": seven,
            "spend": spend_data(rows, days)}
    # "</" would close the <script> block the data is embedded in.
    page = GRAPH_TEMPLATE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(page)
    print(f"  wrote {path}  ({len(five)} five-hour and {len(seven)} weekly windows, "
          f"{len(data['spend']['projects'])} projects over {days} days)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--days", type=int, default=28, help="window to analyse (default 28)")
    parser.add_argument("--sessions", type=int, default=12, help="rows in the session table")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    parser.add_argument("--csv", metavar="PATH", help="also dump every request to a CSV")
    parser.add_argument("--graph", nargs="?", const=QUOTA_GRAPH, metavar="PATH",
                        help="write an HTML page of the recorded 5-hour and weekly quota and "
                             f"of the spend over --days (default {QUOTA_GRAPH}) and exit")
    parser.add_argument("--web", type=float, metavar="PERCENT",
                        help="record the claude.ai share of the week in progress, as shown by "
                             "claude.ai's usage page, and exit")
    args = parser.parse_args()

    if args.web is not None:
        return record_web(args.web)

    if args.graph:
        return write_graph(args.graph, args.days)

    if not os.path.isdir(PROJECTS):
        print(f"{PROJECTS} not found — is Claude Code installed?", file=sys.stderr)
        return 1

    rows = load(args.days)
    budget = budgets(rows, args.days)

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
