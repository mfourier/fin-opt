"""
Command-Line Interface for FinOpt.

Purpose
-------
Provides a user-friendly CLI for running simulations, optimizations,
and generating reports without writing Python code.

Commands
--------
- simulate: Run Monte Carlo simulation with specified parameters
- optimize: Find optimal allocation policy for given goals
- config: Validate and display configuration files
- report: Generate summary reports from simulation results

Example Usage
-------------
    # Run simulation from config file
    $ finopt simulate --config config.json --output results/

    # Optimize allocation policy
    $ finopt optimize --config config.json --goal-file goals.yaml --horizon 36

    # Validate configuration
    $ finopt config validate config.json

    # Show version
    $ finopt --version
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import click
import numpy as np

# Lazy imports for performance
def _import_rich():
    """Lazy import Rich for better startup time."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.progress import Progress, SpinnerColumn, TextColumn
        return Console(), Table, Panel, Progress, SpinnerColumn, TextColumn
    except ImportError:
        return None, None, None, None, None, None


def _get_console():
    """Get Rich console or fallback to basic printing."""
    console, *_ = _import_rich()
    return console


# Version
__version__ = "0.1.0"


@click.group()
@click.version_option(version=__version__, prog_name="finopt")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
@click.pass_context
def main(ctx: click.Context, quiet: bool) -> None:
    """
    FinOpt - Goal-based Portfolio Optimization Framework.

    A tool for simulating and optimizing personal investment strategies
    to achieve financial goals with probabilistic guarantees.

    Use 'finopt COMMAND --help' for command-specific help.
    """
    ctx.ensure_object(dict)
    ctx.obj["quiet"] = quiet
    ctx.obj["console"] = _get_console()


@main.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to model configuration file (JSON)"
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for results (default: ./results/)"
)
@click.option(
    "--horizon", "-T",
    type=int,
    default=24,
    help="Simulation horizon in months (default: 24)"
)
@click.option(
    "--simulations", "-n",
    type=int,
    default=1000,
    help="Number of Monte Carlo simulations (default: 1000)"
)
@click.option(
    "--seed", "-s",
    type=int,
    default=None,
    help="Random seed for reproducibility"
)
@click.option(
    "--allocation",
    type=str,
    default=None,
    help="Allocation policy as comma-separated values (e.g., '0.6,0.4')"
)
@click.pass_context
def simulate(
    ctx: click.Context,
    config: Path,
    output: Optional[Path],
    horizon: int,
    simulations: int,
    seed: Optional[int],
    allocation: Optional[str],
) -> None:
    """
    Run Monte Carlo simulation.

    Loads a model configuration and runs simulations to project
    wealth trajectories over the specified horizon.

    Example:
        finopt simulate -c config.json -T 36 -n 5000 --seed 42
    """
    console = ctx.obj.get("console")
    quiet = ctx.obj.get("quiet", False)

    # Import here to avoid slow startup
    from .serialization import load_model

    if not quiet and console:
        console.print(f"[bold blue]Loading model from {config}...[/bold blue]")

    try:
        model = load_model(config)
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)

    # Parse allocation if provided
    M = len(model.accounts)
    if allocation:
        try:
            X_values = [float(x.strip()) for x in allocation.split(",")]
            if len(X_values) != M:
                click.echo(
                    f"Error: Allocation must have {M} values (got {len(X_values)})",
                    err=True
                )
                sys.exit(1)
            if not np.isclose(sum(X_values), 1.0, atol=1e-6):
                click.echo(
                    f"Error: Allocation must sum to 1.0 (got {sum(X_values):.4f})",
                    err=True
                )
                sys.exit(1)
            X = np.tile(X_values, (horizon, 1))
        except ValueError as e:
            click.echo(f"Error parsing allocation: {e}", err=True)
            sys.exit(1)
    else:
        # Default: equal allocation
        X = np.full((horizon, M), 1.0 / M)

    if not quiet and console:
        console.print(f"[bold]Running {simulations:,} simulations over {horizon} months...[/bold]")

    try:
        result = model.simulate(
            T=horizon,
            n_sims=simulations,
            X=X,
            seed=seed,
            start=date.today().replace(day=1),
        )
    except Exception as e:
        click.echo(f"Error during simulation: {e}", err=True)
        sys.exit(1)

    # Display results
    final_wealth = result.wealth[:, -1, :].sum(axis=1)
    median_wealth = np.median(final_wealth)
    p10 = np.percentile(final_wealth, 10)
    p90 = np.percentile(final_wealth, 90)

    if console and not quiet:
        from rich.table import Table
        from rich.panel import Panel

        table = Table(title="Simulation Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Horizon", f"{horizon} months")
        table.add_row("Simulations", f"{simulations:,}")
        table.add_row("Accounts", f"{M}")
        table.add_row("", "")
        table.add_row("Median Final Wealth", f"${median_wealth:,.0f}")
        table.add_row("10th Percentile", f"${p10:,.0f}")
        table.add_row("90th Percentile", f"${p90:,.0f}")

        console.print(table)
    else:
        click.echo(f"Median Final Wealth: ${median_wealth:,.0f}")
        click.echo(f"10th Percentile: ${p10:,.0f}")
        click.echo(f"90th Percentile: ${p90:,.0f}")

    # Save results if output specified
    if output:
        output.mkdir(parents=True, exist_ok=True)
        result_file = output / "simulation_result.json"

        result_data = {
            "horizon": horizon,
            "n_sims": simulations,
            "seed": seed,
            "allocation": X.tolist(),
            "statistics": {
                "median_wealth": float(median_wealth),
                "p10": float(p10),
                "p90": float(p90),
                "mean": float(np.mean(final_wealth)),
                "std": float(np.std(final_wealth)),
            },
            "wealth_trajectories": result.wealth.tolist(),
        }

        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2)

        if not quiet:
            click.echo(f"Results saved to {result_file}")


@main.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to model configuration file (JSON)"
)
@click.option(
    "--goals", "-g",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to goals configuration file (JSON)"
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file for optimal policy (JSON)"
)
@click.option(
    "--horizon", "-T",
    type=int,
    default=None,
    help="Fixed horizon (if not using goal seeker)"
)
@click.option(
    "--max-horizon",
    type=int,
    default=120,
    help="Maximum horizon for goal seeking (default: 120)"
)
@click.option(
    "--simulations", "-n",
    type=int,
    default=500,
    help="Number of Monte Carlo simulations (default: 500)"
)
@click.option(
    "--seed", "-s",
    type=int,
    default=42,
    help="Random seed for reproducibility (default: 42)"
)
@click.option(
    "--objective",
    type=click.Choice(["balanced", "risky", "conservative"]),
    default="balanced",
    help="Optimization objective (default: balanced)"
)
@click.pass_context
def optimize(
    ctx: click.Context,
    config: Path,
    goals: Path,
    output: Optional[Path],
    horizon: Optional[int],
    max_horizon: int,
    simulations: int,
    seed: int,
    objective: str,
) -> None:
    """
    Optimize allocation policy.

    Finds the optimal allocation policy to achieve specified goals
    using CVaR-based convex optimization.

    Example:
        finopt optimize -c config.json -g goals.json --objective balanced
    """
    console = ctx.obj.get("console")
    quiet = ctx.obj.get("quiet", False)

    # Import here to avoid slow startup
    from .serialization import load_model, save_optimization_result
    from .goals import IntermediateGoal, TerminalGoal, GoalSet
    from .optimization import CVaROptimizer, GoalSeeker

    if not quiet and console:
        console.print(f"[bold blue]Loading model and goals...[/bold blue]")

    try:
        model = load_model(config)
    except Exception as e:
        click.echo(f"Error loading model config: {e}", err=True)
        sys.exit(1)

    # Load goals
    try:
        with open(goals, "r") as f:
            goals_data = json.load(f)
    except Exception as e:
        click.echo(f"Error loading goals file: {e}", err=True)
        sys.exit(1)

    # Parse goals
    start_date = date.today().replace(day=1)
    parsed_goals = []

    for g in goals_data.get("goals", []):
        goal_type = g.get("type", "terminal")
        if goal_type == "terminal":
            parsed_goals.append(TerminalGoal(
                account=g["account"],
                threshold=g["threshold"],
                confidence=g.get("confidence", 0.8),
            ))
        elif goal_type == "intermediate":
            parsed_goals.append(IntermediateGoal(
                month=g["month"],
                account=g["account"],
                threshold=g["threshold"],
                confidence=g.get("confidence", 0.8),
            ))
        else:
            click.echo(f"Warning: Unknown goal type '{goal_type}', skipping", err=True)

    if not parsed_goals:
        click.echo("Error: No valid goals found in goals file", err=True)
        sys.exit(1)

    goal_set = GoalSet(parsed_goals, model.accounts, start_date)

    M = len(model.accounts)
    optimizer = CVaROptimizer(n_accounts=M, objective=objective)

    if not quiet and console:
        console.print(f"[bold]Optimizing with {simulations:,} simulations...[/bold]")

    try:
        if horizon:
            # Fixed horizon optimization
            T = horizon
            # Generate simulation data
            np.random.seed(seed)

            # Get contributions
            A = model.income.contributions(
                months=T,
                n_sims=simulations,
                seed=seed,
                output="array"
            )

            # Get returns
            R = model.returns.generate(T=T, n_sims=simulations, seed=seed)

            # Initial wealth
            W0 = np.array([acc.initial_wealth for acc in model.accounts])

            result = optimizer.solve(
                T=T,
                A=A,
                R=R,
                W0=W0,
                goal_set=goal_set,
            )
        else:
            # Goal seeking - find minimum horizon
            seeker = GoalSeeker(optimizer)

            # This would require model.optimize() method
            click.echo("Goal seeking mode not yet implemented in CLI", err=True)
            click.echo("Please specify --horizon for fixed horizon optimization", err=True)
            sys.exit(1)

    except ImportError:
        click.echo("Error: cvxpy not installed. Install with: pip install cvxpy", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error during optimization: {e}", err=True)
        sys.exit(1)

    # Display results
    if console and not quiet:
        from rich.table import Table

        table = Table(title="Optimization Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Horizon", f"{result.T} months")
        table.add_row("Feasible", "Yes" if result.feasible else "No")
        table.add_row("Solve Time", f"{result.solve_time:.2f}s")
        table.add_row("Objective Value", f"{result.objective_value:,.0f}")

        console.print(table)

        # Show allocation summary
        console.print("\n[bold]Allocation Policy Summary:[/bold]")
        avg_alloc = result.X.mean(axis=0)
        for i, acc in enumerate(model.accounts):
            console.print(f"  {acc.name}: {avg_alloc[i]*100:.1f}%")
    else:
        click.echo(f"Feasible: {result.feasible}")
        click.echo(f"Horizon: {result.T} months")
        click.echo(f"Solve Time: {result.solve_time:.2f}s")

    # Save results
    if output:
        save_optimization_result(result, output)
        if not quiet:
            click.echo(f"Optimal policy saved to {output}")


@main.group()
def config() -> None:
    """
    Configuration management commands.

    Validate, display, and manage model configuration files.
    """
    pass


@config.command("validate")
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def config_validate(ctx: click.Context, config_file: Path) -> None:
    """
    Validate a configuration file.

    Checks that the configuration file is valid JSON and
    conforms to the expected schema.

    Example:
        finopt config validate config.json
    """
    console = ctx.obj.get("console")
    quiet = ctx.obj.get("quiet", False)

    from .serialization import load_model

    try:
        model = load_model(config_file)

        if console and not quiet:
            from rich.panel import Panel

            info = f"""
[bold]Model Configuration Valid[/bold]

[cyan]Income:[/cyan]
  Fixed: {'Yes' if model.income.fixed else 'No'}
  Variable: {'Yes' if model.income.variable else 'No'}

[cyan]Accounts ({len(model.accounts)}):[/cyan]
"""
            for acc in model.accounts:
                params = acc.annual_params
                info += f"  - {acc.name}: {params['return']*100:.1f}% return, {params['volatility']*100:.1f}% vol\n"

            console.print(Panel(info, title="Configuration Summary", border_style="green"))
        else:
            click.echo("Configuration is valid")
            click.echo(f"Accounts: {len(model.accounts)}")

    except Exception as e:
        click.echo(f"Configuration validation failed: {e}", err=True)
        sys.exit(1)


@config.command("show")
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option("--format", "-f", type=click.Choice(["json", "table"]), default="table")
@click.pass_context
def config_show(ctx: click.Context, config_file: Path, format: str) -> None:
    """
    Display configuration details.

    Shows the full configuration in either JSON or table format.

    Example:
        finopt config show config.json --format table
    """
    console = ctx.obj.get("console")

    with open(config_file, "r") as f:
        config_data = json.load(f)

    if format == "json":
        click.echo(json.dumps(config_data, indent=2))
    else:
        if console:
            from rich.table import Table
            from rich import print_json

            # Income table
            income_table = Table(title="Income Configuration")
            income_table.add_column("Type", style="cyan")
            income_table.add_column("Base", justify="right")
            income_table.add_column("Growth", justify="right")

            income = config_data.get("income", {})
            if "fixed" in income:
                fixed = income["fixed"]
                income_table.add_row(
                    "Fixed",
                    f"${fixed.get('base', 0):,.0f}",
                    f"{fixed.get('annual_growth', 0)*100:.1f}%"
                )
            if "variable" in income:
                var = income["variable"]
                income_table.add_row(
                    "Variable",
                    f"${var.get('base', 0):,.0f}",
                    f"σ={var.get('sigma', 0)*100:.1f}%"
                )

            console.print(income_table)

            # Accounts table
            accounts_table = Table(title="Accounts")
            accounts_table.add_column("Name", style="cyan")
            accounts_table.add_column("Annual Return", justify="right")
            accounts_table.add_column("Volatility", justify="right")
            accounts_table.add_column("Initial Wealth", justify="right")

            for acc in config_data.get("accounts", []):
                accounts_table.add_row(
                    acc.get("name", "Unknown"),
                    f"{acc.get('annual_return', 0)*100:.1f}%",
                    f"{acc.get('annual_volatility', 0)*100:.1f}%",
                    f"${acc.get('initial_wealth', 0):,.0f}"
                )

            console.print(accounts_table)
        else:
            click.echo(json.dumps(config_data, indent=2))


@config.command("create")
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option("--template", "-t", type=click.Choice(["basic", "advanced"]), default="basic")
@click.pass_context
def config_create(ctx: click.Context, output_file: Path, template: str) -> None:
    """
    Create a new configuration file from template.

    Generates a starter configuration file that can be customized.

    Example:
        finopt config create my_config.json --template basic
    """
    console = ctx.obj.get("console")
    quiet = ctx.obj.get("quiet", False)

    from .serialization import SCHEMA_VERSION

    if template == "basic":
        config_data = {
            "schema_version": SCHEMA_VERSION,
            "income": {
                "fixed": {
                    "base": 1500000,
                    "annual_growth": 0.03
                },
                "variable": {
                    "base": 0,
                    "sigma": 0.0,
                    "annual_growth": 0.0
                },
                "contribution_rate_fixed": 0.3,
                "contribution_rate_variable": 1.0
            },
            "accounts": [
                {
                    "name": "Conservative",
                    "annual_return": 0.06,
                    "annual_volatility": 0.08,
                    "initial_wealth": 0
                },
                {
                    "name": "Aggressive",
                    "annual_return": 0.12,
                    "annual_volatility": 0.15,
                    "initial_wealth": 0
                }
            ]
        }
    else:  # advanced
        config_data = {
            "schema_version": SCHEMA_VERSION,
            "income": {
                "fixed": {
                    "base": 2000000,
                    "annual_growth": 0.04,
                    "salary_raises": {
                        "12": 0.10,
                        "24": 0.08,
                        "36": 0.05
                    }
                },
                "variable": {
                    "base": 500000,
                    "sigma": 0.20,
                    "annual_growth": 0.02,
                    "seasonality": [0.8, 0.8, 0.9, 1.0, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5],
                    "floor": 0,
                    "cap": 2000000
                },
                "contribution_rate_fixed": 0.25,
                "contribution_rate_variable": 1.0
            },
            "accounts": [
                {
                    "name": "Emergency",
                    "annual_return": 0.04,
                    "annual_volatility": 0.03,
                    "initial_wealth": 1000000
                },
                {
                    "name": "Conservative",
                    "annual_return": 0.08,
                    "annual_volatility": 0.10,
                    "initial_wealth": 0
                },
                {
                    "name": "Aggressive",
                    "annual_return": 0.14,
                    "annual_volatility": 0.18,
                    "initial_wealth": 0
                }
            ],
            "correlation": [
                [1.0, 0.3, 0.2],
                [0.3, 1.0, 0.6],
                [0.2, 0.6, 1.0]
            ]
        }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(config_data, f, indent=2)

    if not quiet:
        if console:
            console.print(f"[green]Created configuration file: {output_file}[/green]")
        else:
            click.echo(f"Created configuration file: {output_file}")


@main.command()
@click.option(
    "--result", "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to simulation result file (JSON)"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["summary", "detailed", "csv"]),
    default="summary",
    help="Output format (default: summary)"
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file (for csv format)"
)
@click.pass_context
def report(
    ctx: click.Context,
    result: Path,
    format: str,
    output: Optional[Path],
) -> None:
    """
    Generate reports from simulation results.

    Creates summary statistics and reports from saved simulation
    or optimization results.

    Example:
        finopt report -r results/simulation_result.json --format detailed
    """
    console = ctx.obj.get("console")
    quiet = ctx.obj.get("quiet", False)

    with open(result, "r") as f:
        data = json.load(f)

    stats = data.get("statistics", {})

    if format == "summary":
        if console and not quiet:
            from rich.table import Table
            from rich.panel import Panel

            table = Table(title="Simulation Summary")
            table.add_column("Statistic", style="cyan")
            table.add_column("Value", style="green", justify="right")

            table.add_row("Horizon", f"{data.get('horizon', 'N/A')} months")
            table.add_row("Simulations", f"{data.get('n_sims', 'N/A'):,}")
            table.add_row("", "")
            table.add_row("Mean Wealth", f"${stats.get('mean', 0):,.0f}")
            table.add_row("Median Wealth", f"${stats.get('median_wealth', 0):,.0f}")
            table.add_row("Std Dev", f"${stats.get('std', 0):,.0f}")
            table.add_row("10th Percentile", f"${stats.get('p10', 0):,.0f}")
            table.add_row("90th Percentile", f"${stats.get('p90', 0):,.0f}")

            console.print(table)
        else:
            click.echo(f"Horizon: {data.get('horizon', 'N/A')} months")
            click.echo(f"Simulations: {data.get('n_sims', 'N/A')}")
            click.echo(f"Mean Wealth: ${stats.get('mean', 0):,.0f}")
            click.echo(f"Median Wealth: ${stats.get('median_wealth', 0):,.0f}")

    elif format == "detailed":
        if console and not quiet:
            from rich.table import Table

            # Basic stats
            click.echo("\n=== Basic Statistics ===")
            for key, value in stats.items():
                click.echo(f"{key}: {value:,.2f}" if isinstance(value, (int, float)) else f"{key}: {value}")

            # Allocation if available
            if "allocation" in data:
                click.echo("\n=== Allocation Policy ===")
                alloc = np.array(data["allocation"])
                avg = alloc.mean(axis=0)
                click.echo(f"Average allocation: {[f'{a*100:.1f}%' for a in avg]}")

        else:
            click.echo(json.dumps(stats, indent=2))

    elif format == "csv":
        import csv

        if not output:
            output = Path("report.csv")

        # Extract wealth trajectories if available
        if "wealth_trajectories" in data:
            trajectories = np.array(data["wealth_trajectories"])
            total_wealth = trajectories.sum(axis=2)  # Sum across accounts

            with open(output, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["simulation"] + [f"month_{t}" for t in range(total_wealth.shape[1])]
                writer.writerow(header)

                for i in range(total_wealth.shape[0]):
                    writer.writerow([i] + list(total_wealth[i]))

            if not quiet:
                click.echo(f"CSV report saved to {output}")
        else:
            click.echo("No wealth trajectories found in result file", err=True)
            sys.exit(1)


@main.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """
    Display system and package information.

    Shows version numbers, installed dependencies, and
    system configuration.
    """
    console = ctx.obj.get("console")

    info_lines = [
        f"FinOpt Version: {__version__}",
        f"Python: {sys.version.split()[0]}",
    ]

    # Check dependencies
    dependencies = {
        "numpy": "numpy",
        "pandas": "pandas",
        "cvxpy": "cvxpy",
        "rich": "rich",
        "click": "click",
    }

    for name, module in dependencies.items():
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "installed")
            info_lines.append(f"{name}: {version}")
        except ImportError:
            info_lines.append(f"{name}: not installed")

    if console:
        from rich.panel import Panel
        console.print(Panel("\n".join(info_lines), title="System Information"))
    else:
        for line in info_lines:
            click.echo(line)


if __name__ == "__main__":
    main()
