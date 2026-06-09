# `cli` — Command-Line Interface

> **Core idea:** Run FinOpt simulations, optimizations, and reports from the
> terminal — no Python scripting required. `cli.py` is a [Click](https://click.palletsprojects.com/)
> application exposed as the `finopt` console script when the package is installed.

---

## Installation

Installing the package registers the `finopt` entry point (see `[project.scripts]` in `pyproject.toml`):

```bash
pip install -e .
finopt --version
finopt --help
```

Every command also runs via `python -m finopt.cli <command>`. Use `finopt COMMAND --help` for command-specific options.

Global options:

| Option | Description |
|--------|-------------|
| `--version` | Print version and exit |
| `-q`, `--quiet` | Suppress non-essential output |

---

## Commands Overview

| Command | Purpose |
|---------|---------|
| `simulate` | Run a Monte Carlo wealth simulation from a config |
| `optimize` | Find the optimal allocation policy for a set of goals |
| `report` | Generate summary/detailed/CSV reports from saved results |
| `config validate` | Validate a configuration file against the schema |
| `config show` | Display a configuration (JSON or table) |
| `config create` | Scaffold a starter configuration from a template |
| `info` | Show version, dependencies, and system information |

---

## `finopt simulate`

Run Monte Carlo simulations to project wealth trajectories over a fixed horizon.

```bash
finopt simulate -c examples/basic_config.json -T 36 -n 5000 --seed 42
```

| Option | Default | Description |
|--------|---------|-------------|
| `-c, --config PATH` | *(required)* | Model configuration file (JSON) |
| `-o, --output PATH` | `./results/` | Output directory for results |
| `-T, --horizon INTEGER` | `24` | Simulation horizon in months |
| `-n, --simulations INTEGER` | `1000` | Number of Monte Carlo paths |
| `-s, --seed INTEGER` | — | Random seed for reproducibility |
| `--allocation TEXT` | — | Allocation policy as comma-separated values (e.g. `'0.6,0.4'`) |

---

## `finopt optimize`

Find the optimal allocation policy to achieve a set of goals via CVaR-based convex optimization. With `--max-horizon` (and no fixed `--horizon`), `GoalSeeker` searches for the minimum feasible horizon $T^\star$.

```bash
finopt optimize -c examples/basic_config.json -g examples/basic_goals.json --objective balanced
```

| Option | Default | Description |
|--------|---------|-------------|
| `-c, --config PATH` | *(required)* | Model configuration file (JSON) |
| `-g, --goals PATH` | *(required)* | Goals configuration file (JSON) |
| `-o, --output PATH` | — | Output file for the optimal policy (JSON) |
| `-T, --horizon INTEGER` | — | Fixed horizon (skips the goal seeker) |
| `--max-horizon INTEGER` | `120` | Maximum horizon for goal seeking |
| `-n, --simulations INTEGER` | `500` | Number of Monte Carlo paths |
| `-s, --seed INTEGER` | `42` | Random seed for reproducibility |
| `--objective [balanced\|risky\|conservative]` | `balanced` | Inner optimization objective |

---

## `finopt report`

Create summary statistics and reports from a saved simulation or optimization result.

```bash
finopt report -r results/simulation_result.json --format detailed
finopt report -r results/simulation_result.json --format csv -o summary.csv
```

| Option | Default | Description |
|--------|---------|-------------|
| `-r, --result PATH` | *(required)* | Result file to summarize (JSON) |
| `-f, --format [summary\|detailed\|csv]` | `summary` | Output format |
| `-o, --output PATH` | — | Output file (used by the `csv` format) |

---

## `finopt config`

Configuration-management subcommands.

### `config validate`

```bash
finopt config validate examples/basic_config.json
```

Checks that the file is valid JSON and conforms to the expected schema (see [Serialization](serialization.md)).

### `config show`

```bash
finopt config show examples/basic_config.json --format table
```

| Option | Default | Description |
|--------|---------|-------------|
| `-f, --format [json\|table]` | `table` | Display format |

### `config create`

```bash
finopt config create my_config.json --template advanced
```

| Option | Default | Description |
|--------|---------|-------------|
| `-t, --template [basic\|advanced]` | `basic` | Starter template |

---

## `finopt info`

Display version numbers, installed dependencies, and system configuration. Useful for bug reports.

```bash
finopt info
```

---

## See also

- [Quick Start](quickstart.md) — the equivalent Python API
- [Configuration](config.md) — the schema behind config files
- [Serialization](serialization.md) — how configs and results are stored
