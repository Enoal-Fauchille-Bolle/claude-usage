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
```

The reports and the HTML page contain project names and paths, so check them before sharing. Accounts appear under the name of their config directory (`.claude`, `.claude2`).

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
| `~/.claude/state/quota-history.jsonl` | official `/usage` percentages, written by a status line (see below) |
| `~/.claude/state/web-share.jsonl` | claude.ai share of each week, written by `--web` |
| `~/.claude/state/quota-graph.html` | default output of `--graph` |

### Quota history format

Claude Code does not keep a history of its `/usage` percentages, so a custom status line has to record them. The history is optional: without it, `--graph` and `--web` stop with a message and the report uses the built-in budgets.

Each line of `quota-history.jsonl` is one JSON object, appended when a value changes:

```json
{"ts":1793477750,"account":".claude","five":28,"seven":57,"five_reset":1793479200,"seven_reset":1793714400}
```

| Field | Meaning | Status line source |
| --- | --- | --- |
| `ts` | time of the reading, Unix seconds | current time |
| `account` | name of the config directory | basename of `CLAUDE_CONFIG_DIR`, or `.claude` |
| `five` | percentage used of the 5-hour window | `.rate_limits.five_hour.used_percentage` |
| `seven` | percentage used of the week | `.rate_limits.seven_day.used_percentage` |
| `five_reset` | end of the 5-hour window, Unix seconds | `.rate_limits.five_hour.resets_at` |
| `seven_reset` | end of the week, Unix seconds | `.rate_limits.seven_day.resets_at` |

`five`, `seven` and their reset times may be `null`. Readings with an impossible percentage or reset time are ignored, and so is any reading lower than the previous one in the same window, because several sessions can write at once. Only the account of the most recent reading is used.

A working recorder is in [`statusline-command.sh`](https://github.com/Enoal-Fauchille-Bolle/Dotfiles/blob/main/home/claude/.claude/statusline-command.sh).

## Development

```sh
uv run pytest
```

## License

GPL-3.0, see [LICENSE](LICENSE).
