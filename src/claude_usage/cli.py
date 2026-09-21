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
    claude-usage --graph         # HTML line chart of the official 5-hour/weekly quota
    claude-usage --graph out.html
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
import time

# --- Calibration -----------------------------------------------------------
# Measured on a Pro plan by cross-checking this script against /usage:
#   "Current session: 44% used"  <-> $28.37 of equivalent spend  -> ~$64 per 5h
#   "Current week: 38% used"     <-> $125 of equivalent spend    -> ~$330 per week
# The weekly figure is a re-measure from September: the August one came from a
# 5% reading, and rounding error is multiplied by 20 at that size. It matches a
# second reading (main account, 25% with about $40 of Claude Code in it). It
# includes the +25% weekly bonus granted permanently since 2026-09-14, so it is
# not divided by anything. Only Claude Code on this machine is counted: claude.ai
# shares the same limit and is invisible here.
SESSION_BUDGET = 64.0       # $ equivalent per 5-hour window, one Pro seat
WEEKLY_BUDGET = 330.0       # $ equivalent per week, one Pro seat

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

# The status line appends one reading of the official /usage percentages here
# each time they change (see claude/.claude/statusline-command.sh).
QUOTA_HISTORY = os.path.expanduser("~/.claude/state/quota-history.jsonl")
QUOTA_GRAPH = os.path.expanduser("~/.claude/state/quota-graph.html")
QUOTA_SPAN = {"five": 5 * 3600, "seven": 7 * 86400}  # window length, seconds


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


GRAPH_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Claude quota history</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --hover: rgba(11,11,11,0.05);
  --s1: #2a78d6; --s2: #eb6834;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --hover: rgba(255,255,255,0.08);
    --s1: #3987e5; --s2: #d95926;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --hover: rgba(255,255,255,0.08);
  --s1: #3987e5; --s2: #d95926;
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
@media (forced-colors: active) { .line.s2 { stroke-dasharray: 6 4; } }
</style>
</head>
<body>
<main>
  <h1>Claude quota history</h1>
  <p class="sub" id="sub"></p>
  <div class="card">
    <div class="filters" id="filters" role="group" aria-label="Time range"></div>
    <div id="charts"></div>
    <p class="note">Each line is one limit window; a break means the window ended or nothing was recorded.
      Readings below an earlier one in the same window are stale and ignored.</p>
    <details><summary>Table view</summary><div id="tables"></div></details>
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

document.getElementById("sub").textContent =
  "Official /usage percentages recorded by the status line" + (D.account ? " · account " + D.account : "");
tables();
render();
let drawnWidth = document.getElementById("charts").clientWidth;
new ResizeObserver(() => {
  const w = document.getElementById("charts").clientWidth;
  if (w !== drawnWidth) { drawnWidth = w; render(); }
}).observe(document.getElementById("charts"));
})();
</script>
</body>
</html>
"""


def write_graph(path: str) -> int:
    """Render the quota history as a self-contained HTML page (no external assets)."""
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

    data = {"now": int(time.time()), "account": account, "five": five, "seven": seven}
    # "</" would close the <script> block the data is embedded in.
    page = GRAPH_TEMPLATE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(page)
    print(f"  wrote {path}  ({len(five)} five-hour and {len(seven)} weekly windows)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--days", type=int, default=28, help="window to analyse (default 28)")
    parser.add_argument("--sessions", type=int, default=12, help="rows in the session table")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    parser.add_argument("--csv", metavar="PATH", help="also dump every request to a CSV")
    parser.add_argument("--graph", nargs="?", const=QUOTA_GRAPH, metavar="PATH",
                        help="write an HTML chart of the recorded 5-hour and weekly quota "
                             f"(default {QUOTA_GRAPH}) and exit")
    args = parser.parse_args()

    if args.graph:
        return write_graph(args.graph)

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
