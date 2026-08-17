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
be useful. The two calibration constants below were measured by comparing this
script's output against the percentages reported by /usage on a Pro plan.

Usage:
    claude-usage                 # last 28 days
    claude-usage --days 7        # shorter window
    claude-usage --sessions 20   # show more sessions in the session table
    claude-usage --json          # machine-readable output
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
import sys

# --- Calibration -----------------------------------------------------------
# Measured on a Pro plan by cross-checking this script against /usage:
#   "Current session: 44% used"  <-> $28.37 of equivalent spend  -> ~$64 per 5h
#   "Current week: 5% used"      <-> $82.04 of equivalent spend  -> ~$1640 per week
# The weekly figure was taken while the +50% Claude Code weekly promo was
# active, so the standard weekly budget is that number divided by 1.5.
SESSION_BUDGET = 64.0       # $ equivalent per 5-hour window, one Pro seat
WEEKLY_BUDGET = 1640.0 / 1.5  # $ equivalent per week, one Pro seat, no promo

# Anthropic list prices, $ per million tokens (input, output).
PRICES = {
    "claude-opus-5": (5, 25),
    "claude-opus-4-8": (5, 25),
    "claude-opus-4-7": (5, 25),
    "claude-opus-4-6": (5, 25),
    "claude-sonnet-5": (3, 15),
    "claude-sonnet-4-6": (3, 15),
    "claude-haiku-4-5-20251001": (1, 5),
}

# Cache writes cost more than fresh input, cache reads cost far less.
CACHE_WRITE_5M = 1.25
CACHE_WRITE_1H = 2.0
CACHE_READ = 0.1

# A request sent with more context than this is flagged as "long context".
LONG_CONTEXT = 150_000
HUGE_CONTEXT = 300_000

# A gap longer than the 1-hour cache lifetime means the next request re-writes
# the whole context at cache-write price instead of reading it at 0.1x.
CACHE_LIFETIME_H = 1.0

PROJECTS = os.path.expanduser("~/.claude/projects")


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
    """
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    seen: dict[str, dict] = {}

    for path in glob.glob(os.path.join(PROJECTS, "*", "*.jsonl")):
        project = os.path.basename(os.path.dirname(path))
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
                    continue
                rate_in, rate_out = price
                cost = (
                    inp * rate_in
                    + write_5m * rate_in * CACHE_WRITE_5M
                    + write_1h * rate_in * CACHE_WRITE_1H
                    + read * rate_in * CACHE_READ
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

    return sorted(seen.values(), key=lambda r: r["t"])


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


def report(rows: list[dict], days: int, n_sessions: int) -> None:
    if not rows:
        print("No transcripts found in the requested window.", file=sys.stderr)
        return

    total = sum(r["cost"] for r in rows)
    long_cost = sum(r["cost"] for r in rows if r["context"] > LONG_CONTEXT)
    huge_cost = sum(r["cost"] for r in rows if r["context"] > HUGE_CONTEXT)
    sess = sessions(rows)
    stale = sum(s["stale_cost"] for s in sess)

    print(f"\n  {len(rows):,} requests over {days} days — ${total:,.0f} equivalent\n")

    print("  WEEKLY  (one Pro seat ≈ ${:,.0f}/week)".format(WEEKLY_BUDGET))
    for monday, spend in weekly(rows):
        pct = spend / WEEKLY_BUDGET * 100
        bar = "█" * min(40, round(pct / 2.5))
        print(f"    {monday}  ${spend:7.0f}  {pct:5.1f}%  {bar}")

    print(f"\n  DAILY 5-HOUR PEAK  (limit ≈ ${SESSION_BUDGET:.0f})")
    over = 0
    for day, peak, day_total in daily_peaks(rows):
        hit = peak > SESSION_BUDGET
        over += hit
        bar = "█" * min(30, round(peak / SESSION_BUDGET * 20))
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
              f"{s['stale_cost']:7.1f}  {s['project'].split('-')[-1]}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--days", type=int, default=28, help="window to analyse (default 28)")
    parser.add_argument("--sessions", type=int, default=12, help="rows in the session table")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    parser.add_argument("--csv", metavar="PATH", help="also dump every request to a CSV")
    args = parser.parse_args()

    if not os.path.isdir(PROJECTS):
        print(f"{PROJECTS} not found — is Claude Code installed?", file=sys.stderr)
        return 1

    rows = load(args.days)

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
            "weekly": [{"week": str(w), "cost": c} for w, c in weekly(rows)],
            "daily": [{"day": str(d), "peak_5h": p, "total": t} for d, p, t in daily_peaks(rows)],
            "sessions": [{**s, "start": s["start"].isoformat()} for s in sessions(rows)],
        }, sys.stdout, indent=2)
        return 0

    report(rows, args.days, args.sessions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
