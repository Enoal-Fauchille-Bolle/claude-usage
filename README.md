# claude-usage

Reports Claude Code plan-limit consumption from the local transcripts: how close the weekly and 5-hour limits are, which sessions spend the most on re-read context, and what stale sessions cost in cache misses.

Amounts are in "API list price equivalent" dollars. A subscription is a flat fee, so these dollars are not billed, but they track plan-limit consumption closely enough to compare sessions and weeks.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13 or later.

```sh
uv tool install git+https://github.com/Enoal-Fauchille-Bolle/claude-usage
```

Update with `uv tool upgrade claude-usage`.

## Usage

```sh
claude-usage                      # report on the last 28 days
claude-usage --days 7             # shorter window
claude-usage --sessions 20        # more rows in the session table
claude-usage --json               # machine-readable output
claude-usage --csv requests.csv   # also dump every request to a CSV
claude-usage --graph              # HTML page of the quota and the spend
claude-usage --web 22             # claude.ai took 22% of this week's usage
claude-usage record               # run by the status line, see "Recording the quota"
claude-usage guard                # run by a hook, see "Warnings before a request"
```

The reports and the HTML page contain project names and paths, so check them before sharing. Accounts appear under the name of their config directory (`.claude`, `.claude2`).

## Recording the quota

Claude Code keeps no history of its `/usage` percentages and hands them to the status line only. `claude-usage record` reads what the status line receives on stdin, appends the percentages to the quota history when one of them changes, and saves the context size and the cost for `claude-usage guard`. It prints nothing.

Without a status line, add this to `~/.claude/settings.json`. The status line stays blank:

```json
{
  "statusLine": { "type": "command", "command": "claude-usage record" }
}
```

With a status line script of your own, add this after the script has read its input:

```sh
input=$(cat)
printf '%s' "$input" | claude-usage record >/dev/null 2>&1 &
```

The `&` keeps the display from waiting for Python to start. The redirection is required with it, because a background command that inherits the script's output can hold the status line back and print into it.

Claude Code may start the status line from an environment without `~/.local/bin` in its `PATH`, a desktop launcher for example. If the history stops growing, call the command by its full path.

## Warnings before a request

Claude cannot see how full its own context is. `claude-usage guard` is a `UserPromptSubmit` hook, the event whose output reaches Claude before the request is sent, and it prints two lines when a threshold is crossed:

```text
[budget] context 320k (32%) · $16.00 · 5h 73% (pace 80%, -7) · 7d 75% (pace 85%, -10)
[budget] thresholds crossed: CONTEXT_STOP,COST
```

Below every threshold it prints nothing, so most prompts cost no tokens. To enable it, add this to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [{ "type": "command", "command": "claude-usage guard" }] }
    ]
  }
}
```

The quotas and the cost come from `claude-usage record`, which has to be set up as well. Without it, or when its figures are more than 15 minutes old, only the context size is known, read from the transcript.

The guard names the thresholds and leaves the response to you, so say in your `CLAUDE.md` what Claude should do about each name. For example: `CONTEXT_DECISION: propose /compact for the same task, /clear for a new one.`

| Threshold | Crossed when | Environment variable | Default |
| --- | --- | --- | --- |
| `CONTEXT_DECISION` | the context reaches this many tokens | `CLAUDE_USAGE_CONTEXT_DECISION` | 150000 |
| `CONTEXT_STOP` | the context reaches this many tokens, replacing `CONTEXT_DECISION` | `CLAUDE_USAGE_CONTEXT_STOP` | 300000 |
| `COST` | the conversation reaches this many dollars | `CLAUDE_USAGE_COST_WARN` | 15 |
| `QUOTA_5H`, `QUOTA_7D` | usage is more than this many points ahead of the pace | `CLAUDE_USAGE_PACE_GAP` | 10 |
| `QUOTA_5H`, `QUOTA_7D` | usage reaches this percentage, whatever the pace | `CLAUDE_USAGE_QUOTA_HARD` | 90 |
| `QUOTA_5H` | usage reaches this percentage and the reset time is unknown | `CLAUDE_USAGE_FIVE_WARN` | 70 |
| `QUOTA_7D` | usage reaches this percentage and the reset time is unknown | `CLAUDE_USAGE_SEVEN_WARN` | 80 |
| `COLD_CACHE` | the prompt cache has expired and the context has at least this many tokens | `CLAUDE_USAGE_CACHE_MIN` | 100000 |

The pace is the share of the window already elapsed: 73% used with 80% of the window gone is 7 points of margin. Two more variables adjust the measures: `CLAUDE_USAGE_CACHE_TTL` is the prompt cache lifetime in seconds (3600, or 300 when drawing on usage credits), and `CLAUDE_USAGE_STATE_MAX_AGE` is the age in seconds past which the figures from `record` are ignored (900). An invalid value falls back on the default.

The hook never exits with 2, the code that makes Claude Code reject the prompt.

## Budgets

A budget is the dollar equivalent of a full window. The tool picks each one in this order:

1. The value given by a flag or an environment variable.
2. The value measured from the quota history, when it holds enough readings.
3. A built-in constant, calibrated on one Pro account: $40 per 5-hour window and $270 per week.

| Budget | Flag | Environment variable |
| --- | --- | --- |
| 5-hour window | `--session-budget` | `CLAUDE_USAGE_SESSION_BUDGET` |
| Week | `--weekly-budget` | `CLAUDE_USAGE_WEEKLY_BUDGET` |

Another plan needs its own values while the history is too thin to measure them.

## Files read and written

| Path | Use |
| --- | --- |
| `~/.claude/projects/` | Claude Code transcripts, read only |
| `~/.claude*/history.jsonl` | session IDs of each config directory, to split spend by account |
| `~/.claude/state/quota-history.jsonl` | official `/usage` percentages, written by `record` |
| `~/.claude/state/quota-last-<account>.txt` | last recorded values, so that `record` only appends a change |
| `~/.claude/state/ctx-<session>.txt` | latest figures of a session, written by `record` for `guard`, deleted after 7 days |
| `~/.claude/state/web-share.jsonl` | claude.ai share of each week, written by `--web` |
| `~/.claude/state/quota-graph.html` | default output of `--graph` |

### Quota history format

The history is optional: without it, `--graph` and `--web` stop with a message and the report uses the built-in budgets.

Each line of `quota-history.jsonl` is one JSON object, appended when a value changes:

```json
{"ts":1793477750,"account":".claude","five":28,"seven":57,"five_reset":1793479200,"seven_reset":1793714400}
```

| Field | Meaning | Source |
| --- | --- | --- |
| `ts` | time of the reading, Unix seconds | current time |
| `account` | name of the config directory | basename of `CLAUDE_CONFIG_DIR`, or `.claude` |
| `five` | percentage used of the 5-hour window | `.rate_limits.five_hour.used_percentage` |
| `seven` | percentage used of the week | `.rate_limits.seven_day.used_percentage` |
| `five_reset` | end of the 5-hour window, Unix seconds | `.rate_limits.five_hour.resets_at` |
| `seven_reset` | end of the week, Unix seconds | `.rate_limits.seven_day.resets_at` |

`five`, `seven` and their reset times may be `null`. Readings with an impossible percentage or reset time are ignored, and so is any reading lower than the previous one in the same window, because several sessions can write at once. Only the account of the most recent reading is used.

## Development

```sh
uv run pytest
```

## License

GPL-3.0, see [LICENSE](LICENSE).
