#!python3
"""Orchestrate the execution of the benchmarking targets and consolidate all of
their Google-Benchmark JSON reports into a single, tidy CSV file (plus optional
plots).

Two-step pipeline:

1. *Benchmark*: search the given ``--path`` directories for **executables** whose
   path matches one of the ``--regex`` patterns and run each of them, producing a
   ``<executable>.json`` report next to the current working directory.
2. *Consolidate*: load every report, normalize it into a tidy DataFrame and write
   a single unified CSV.

With ``--skip-benchmark`` the first step is skipped and ``--path`` / ``--regex``
are used to locate the **report files** directly.

The working directory is always assumed to be the build folder (configurable via
``--build-dir``); benchmark binaries frequently load auxiliary data relative to
it, so we ``chdir`` there before doing anything else.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.colors import LogNorm

# --------------------------------------------------------------------------- #
# Column names of the unified output CSV
# --------------------------------------------------------------------------- #
BENCHMARK_PROBLEM = "Benchmark Problem"
PARADIGM = "Paradigm"
DESCRIPTION = "Description"
PRECISION = "Precision"
HARDWARE = "Hardware"
PROBLEM_SIZE = "Problem Size"
ITERATIONS = "Iterations"
WALL_CLOCK_TIME = "Wall Clock Time"
CPU_TIME = "CPU Time"
KERNEL_TIME = "Kernel Time"
NEIGHBOR_SEARCH_TIME = "Neighbor Search Time"
POSITION_UPDATE_TIME = "Position Update Time"
VELOCITY_UPDATE_TIME = "Velocity Update Time"
FORCE_UPDATE_TIME = "Force Update Time"
TIME_UNIT = "Time Unit"

COLUMN_LIST = [
    BENCHMARK_PROBLEM,
    PARADIGM,
    DESCRIPTION,
    PRECISION,
    HARDWARE,
    PROBLEM_SIZE,
    ITERATIONS,
    WALL_CLOCK_TIME,
    CPU_TIME,
    KERNEL_TIME,
    NEIGHBOR_SEARCH_TIME,
    POSITION_UPDATE_TIME,
    VELOCITY_UPDATE_TIME,
    FORCE_UPDATE_TIME,
    TIME_UNIT,
]
"""All columns in the output CSV file, in order."""

# Map the raw Google-Benchmark JSON keys to our column names. Reports only carry
# a subset of these (e.g. the n-body report has the per-phase timers, matMul has
# a kernel timer); missing ones are filled with NA.
RAW_KEY_TO_COLUMN = {
    "real_time": WALL_CLOCK_TIME,
    "cpu_time": CPU_TIME,
    "time_unit": TIME_UNIT,
    "iterations": ITERATIONS,
    "kernel_time": KERNEL_TIME,
    "neighbor_search": NEIGHBOR_SEARCH_TIME,
    "position_update_reset": POSITION_UPDATE_TIME,
    "velocity_update": VELOCITY_UPDATE_TIME,
    "force_update": FORCE_UPDATE_TIME,
}

# Numeric runtime columns (used for unit conversion and plotting).
RUNTIME_COLUMNS = [
    WALL_CLOCK_TIME,
    CPU_TIME,
    KERNEL_TIME,
    NEIGHBOR_SEARCH_TIME,
    POSITION_UPDATE_TIME,
    VELOCITY_UPDATE_TIME,
    FORCE_UPDATE_TIME,
]

# User-facing runtime choices -> column name.
RUNTIME_CHOICES = {
    "wall_time": WALL_CLOCK_TIME,
    "cpu_time": CPU_TIME,
    "kernel_time": KERNEL_TIME,
}

MERGED_PARADIGM_DESCRIPTION = "Paradigm[Description]"
"""Helper column merging paradigm + description for plotting."""

# Conversion factors from a given time unit to nanoseconds.
TIME_UNIT_TO_NS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def find_files(
    search_dirs: list[Path],
    patterns: list[str],
    require_executable: bool = False,
    exclude: list[str] | None = None,
) -> list[Path]:
    """Recursively search ``search_dirs`` for files whose path matches any of the
    given regular expressions.

    Args:
        search_dirs: Directories to search recursively.
        patterns: Regular expression patterns. A file is kept if any pattern
            matches (``re.search``) against its path.
        require_executable: If True, only keep files with the executable bit set
            (used to locate benchmark binaries).
        exclude: Regular expression patterns. A file is dropped if any of these
            matches (``re.search``) against its path, even if it matched a
            include pattern (e.g. ``".*cpp.*"``).

    Returns:
        A sorted, de-duplicated list of resolved matching file paths.
    """
    compiled = [re.compile(p) for p in patterns]
    excluded = [re.compile(p) for p in (exclude or [])]
    kind = "executables" if require_executable else "files"
    logger.info(f"Searching {search_dirs} for {kind} matching: {patterns}")
    if excluded:
        logger.info(f"Excluding {kind} matching: {exclude}")

    matches: set[Path] = set()
    for directory in search_dirs:
        if not directory.exists():
            logger.warning(f"Search path {directory} does not exist, skipping")
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if require_executable and not os.access(path, os.X_OK):
                continue
            path_str = str(path)
            if not any(rx.search(path_str) for rx in compiled):
                continue
            if any(rx.search(path_str) for rx in excluded):
                logger.debug(f"Excluded {kind[:-1]}: {path}")
                continue
            resolved = path.resolve()
            if resolved not in matches:
                logger.debug(f"Matched {kind[:-1]}: {path}")
                matches.add(resolved)

    result = sorted(matches)
    logger.info(f"Found {len(result)} matching {kind}")
    return result


def print_found_files(files: list[Path], base: Path, are_reports: bool) -> None:
    """Pretty-print the discovered files as a boxed, numbered table (dry run)."""
    kind = "report(s)" if are_reports else "executable(s)"
    rows = []
    for i, path in enumerate(files, start=1):
        try:
            shown = path.relative_to(base)
        except ValueError:
            shown = path
        rows.append((str(i), str(shown)))

    title = f" Found {len(files)} {kind} "
    idx_w = max((len(r[0]) for r in rows), default=1)
    path_w = max((len(r[1]) for r in rows), default=0)
    inner = max(idx_w + path_w + 3, len(title))

    top = f"┌{'─' * inner}┐"
    sep = f"├{'─' * inner}┤"
    bottom = f"└{'─' * inner}┘"

    print(top)
    print(f"│{title.center(inner)}│")
    print(sep)
    if rows:
        for num, shown in rows:
            line = f" {num.rjust(idx_w)}  {shown.ljust(path_w)} "
            print(f"│{line.ljust(inner)}│")
    else:
        print(f"│{' (nothing matched) '.center(inner)}│")
    print(bottom)


def run_benchmarks(executables: list[Path], force: bool = False) -> list[Path]:
    """Run each benchmark executable, producing a ``<name>.json`` Google-Benchmark
    report in the current working directory.

    Args:
        executables: Benchmark binaries to run.
        force: Re-run even if a report already exists (otherwise it is reused).

    Returns:
        The list of report files that exist after the run.
    """
    report_files: list[Path] = []
    for target in executables:
        output_file = Path(f"{target.name}.json")
        if output_file.exists() and not force:
            logger.warning(
                f"Report {output_file} already exists, reusing it (use --force to re-run)"
            )
            report_files.append(output_file)
            continue
        try:
            logger.info(f"Benchmarking {target.name} ...")
            subprocess.run(
                [
                    str(target),
                    f"--benchmark_out={output_file.name}",
                    "--benchmark_out_format=json",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.success(f"Finished {target.name} -> {output_file}")
            report_files.append(output_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to execute {target}: {e}")
    return report_files


# --------------------------------------------------------------------------- #
# Loading / normalization
# --------------------------------------------------------------------------- #
def _normalize_report(json_data: dict, hardware: str) -> pd.DataFrame:
    """Turn a single parsed Google-Benchmark report into a tidy DataFrame."""
    context = json_data["context"]
    paradigm = context.get("paradigm", "Unknown")
    float_type = str(context.get("float_type", "Unknown"))

    df = pd.DataFrame(json_data["benchmarks"])
    if df.empty:
        return df

    # Drop aggregate rows (BigO / RMS / mean / stddev ...); we only want raw runs.
    if "run_type" in df.columns:
        df = df[df["run_type"] == "iteration"].copy()
    if df.empty:
        return df

    # --- Parse the encoded name -------------------------------------------- #
    # Convention: "<BenchmarkProblem>[-<Description>][/<ProblemSize>]"
    name_size = df["name"].str.split("/", expand=True)
    base = name_size[0]
    has_size_in_name = name_size.shape[1] > 1

    details = base.str.split("-", n=1, expand=True)
    df[BENCHMARK_PROBLEM] = details[0].astype(str)
    df[DESCRIPTION] = (
        details[1].fillna("").astype(str) if details.shape[1] > 1 else ""
    )

    # --- Problem size ------------------------------------------------------- #
    if has_size_in_name:
        df[PROBLEM_SIZE] = pd.to_numeric(name_size[1], errors="coerce")
    elif "NumFaces" in df.columns:
        # The polyhedral benchmark has no size in the name; it reports the number
        # of faces of the processed mesh as a user counter instead.
        df[PROBLEM_SIZE] = pd.to_numeric(df["NumFaces"], errors="coerce")
    else:
        df[PROBLEM_SIZE] = pd.NA

    # --- Context columns ---------------------------------------------------- #
    df[PARADIGM] = paradigm
    df[PRECISION] = float_type
    df[HARDWARE] = hardware

    # --- Rename the runtime/counter columns we know about ------------------- #
    df = df.rename(columns={k: v for k, v in RAW_KEY_TO_COLUMN.items() if k in df})

    # Ensure every output column exists, then select & order them.
    for col in COLUMN_LIST:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[COLUMN_LIST]

    # Coerce numeric columns.
    for col in [PROBLEM_SIZE, ITERATIONS, *RUNTIME_COLUMNS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_reports(report_files: list[Path], hardware: str = "") -> pd.DataFrame:
    """Load and normalize a list of Google-Benchmark JSON reports.

    Args:
        report_files: Paths to JSON report files.
        hardware: Free-form hardware identifier stored in the ``Hardware`` column
            (e.g. ``"RTX5080"``).

    Returns:
        A single tidy DataFrame (one row per benchmark run).
    """
    frames: list[pd.DataFrame] = []
    logger.info(f"Loading {len(report_files)} report file(s)")
    for file_path in report_files:
        logger.debug(f"Loading {file_path}")
        try:
            with open(file_path, "r") as fh:
                json_data = json.load(fh)
        except (ValueError, OSError) as e:
            logger.error(f"Could not read {file_path}: {e}")
            continue
        df = _normalize_report(json_data, hardware)
        if df.empty:
            logger.warning(f"No usable benchmark rows in {file_path}")
            continue
        frames.append(df)

    if not frames:
        logger.error("No benchmark data could be loaded.")
        return pd.DataFrame(columns=COLUMN_LIST)

    combined = pd.concat(frames, ignore_index=True)
    logger.success(
        f"Loaded {len(combined)} rows across "
        f"{combined[BENCHMARK_PROBLEM].nunique()} problem(s) and "
        f"{combined[PARADIGM].nunique()} paradigm(s)"
    )
    return combined


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def _add_merged_paradigm(df: pd.DataFrame) -> str:
    """Add the merged ``Paradigm[Description]`` column and return its name."""
    df[MERGED_PARADIGM_DESCRIPTION] = np.where(
        df[DESCRIPTION].fillna("") == "",
        df[PARADIGM],
        df[PARADIGM] + "[" + df[DESCRIPTION].astype(str) + "]",
    )
    return MERGED_PARADIGM_DESCRIPTION


def _convert_runtime(df: pd.DataFrame, column: str, target_unit: str) -> pd.Series:
    """Return ``column`` converted from each row's ``Time Unit`` to ``target_unit``."""
    target_factor = TIME_UNIT_TO_NS.get(target_unit)
    if target_factor is None:
        logger.warning(f"Unknown time unit '{target_unit}', leaving values as-is")
        return df[column]
    src_factor = df[TIME_UNIT].map(TIME_UNIT_TO_NS).fillna(1.0)
    return df[column] * src_factor / target_factor


def _save(fig_or_grid, save_path: Path, suffix: str) -> None:
    """Save a figure/FacetGrid as both PDF (vector) and PNG (preview)."""
    for ext in (".pdf", ".png"):
        out = save_path.with_name(f"{save_path.stem}_{suffix}{ext}")
        fig_or_grid.savefig(out, dpi=300, bbox_inches="tight")
        logger.success(f"Saved {out}")


def _prepare_plot_df(
    df: pd.DataFrame,
    selected_runtime: str,
    time_unit: str,
    use_merged: bool,
) -> tuple[pd.DataFrame, str, str]:
    """Common preprocessing for all charts.

    Returns ``(df, runtime_column, hue_column)``.
    """
    df = df.copy()
    runtime = RUNTIME_CHOICES[selected_runtime]
    df = df.dropna(subset=[runtime])
    df[runtime] = _convert_runtime(df, runtime, time_unit)
    hue = _add_merged_paradigm(df) if use_merged else PARADIGM
    return df, runtime, hue


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def plot_linechart(
    benchmark_df: pd.DataFrame,
    save_path: Path | None = None,
    time_unit: str = "ns",
    selected_runtime: Literal["kernel_time", "wall_time", "cpu_time"] = "wall_time",
    use_merged_paradigm_description: bool = False,
) -> sns.FacetGrid:
    """Log-log runtime-vs-problem-size scaling plot, one facet per benchmark
    problem, one line per paradigm and line style per precision.
    """
    df, runtime, hue = _prepare_plot_df(
        benchmark_df, selected_runtime, time_unit, use_merged_paradigm_description
    )
    df = df.dropna(subset=[PROBLEM_SIZE]).sort_values(PROBLEM_SIZE)

    n_problems = df[BENCHMARK_PROBLEM].nunique()
    n_precision = df[PRECISION].nunique()
    grid = sns.relplot(
        data=df,
        x=PROBLEM_SIZE,
        y=runtime,
        hue=hue,
        style=PRECISION if n_precision > 1 else None,
        col=BENCHMARK_PROBLEM,
        col_wrap=min(n_problems, 3),
        kind="line",
        marker="o",
        palette="tab10",
        facet_kws={"sharex": False, "sharey": False},
        height=4.5,
        aspect=1.2,
    )
    grid.set(xscale="log", yscale="log")
    grid.set_axis_labels("Problem Size $N$ [1]", f"{runtime} [{time_unit}]")
    grid.set_titles("{col_name}")
    for ax in grid.axes.flat:
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
    grid.figure.suptitle(
        f"Runtime scaling ({runtime}, {time_unit})", y=1.02, fontsize=14
    )
    if save_path is not None:
        _save(grid, save_path, "line")
    return grid


def plot_barchart(
    benchmark_df: pd.DataFrame,
    save_path: Path | None = None,
    time_unit: str = "ns",
    selected_runtime: Literal["kernel_time", "wall_time", "cpu_time"] = "wall_time",
    use_merged_paradigm_description: bool = False,
) -> sns.FacetGrid:
    """Grouped bar chart comparing paradigms per problem size, one facet per
    benchmark problem.
    """
    df, runtime, hue = _prepare_plot_df(
        benchmark_df, selected_runtime, time_unit, use_merged_paradigm_description
    )
    # Discrete, numerically ordered colour categories for the problem sizes.
    size_col = "Problem Size (cat)"
    sizes = sorted(df[PROBLEM_SIZE].dropna().unique())
    df[size_col] = pd.Categorical(
        df[PROBLEM_SIZE].map(lambda s: f"{int(s)}" if pd.notna(s) else "n/a"),
        categories=[f"{int(s)}" for s in sizes],
        ordered=True,
    )
    n_problems = df[BENCHMARK_PROBLEM].nunique()
    grid = sns.catplot(
        data=df,
        x=hue,
        y=runtime,
        hue=size_col,
        col=BENCHMARK_PROBLEM,
        col_wrap=min(n_problems, 3),
        kind="bar",
        palette="viridis",
        errorbar=None,
        sharex=False,
        sharey=False,
        height=4.5,
        aspect=1.3,
        legend_out=True,
    )
    grid.set_axis_labels("Paradigm", f"{runtime} [{time_unit}]")
    grid.set_titles("{col_name}")
    grid.set(yscale="log")
    if grid.legend is not None:
        grid.legend.set_title("Problem Size")
    for ax in grid.axes.flat:
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        for label in ax.get_xticklabels():
            label.set_rotation(20)
            label.set_ha("right")
    grid.figure.suptitle(
        f"Runtime comparison ({runtime}, {time_unit})", y=1.02, fontsize=14
    )
    if save_path is not None:
        _save(grid, save_path, "bar")
    return grid


def plot_heatmap(
    benchmark_df: pd.DataFrame,
    save_path: Path | None = None,
    time_unit: str = "ns",
    selected_runtime: Literal["kernel_time", "wall_time", "cpu_time"] = "wall_time",
    use_merged_paradigm_description: bool = False,
) -> plt.Figure:
    """Heatmap of runtime (paradigm x problem size) per benchmark problem with a
    logarithmic colour scale. Handy to spot which paradigm wins where at a glance.
    """
    df, runtime, hue = _prepare_plot_df(
        benchmark_df, selected_runtime, time_unit, use_merged_paradigm_description
    )
    df = df.dropna(subset=[PROBLEM_SIZE])
    problems = sorted(df[BENCHMARK_PROBLEM].unique())
    n = len(problems)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False
    )
    for idx, problem in enumerate(problems):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df[BENCHMARK_PROBLEM] == problem]
        pivot = sub.pivot_table(
            index=hue, columns=PROBLEM_SIZE, values=runtime, aggfunc="mean"
        )
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="rocket_r",
            norm=LogNorm(),
            annot=True,
            fmt=".2g",
            linewidths=0.5,
            cbar_kws={"label": f"{runtime} [{time_unit}]"},
        )
        ax.set_title(problem)
        ax.set_xlabel("Problem Size $N$")
        ax.set_ylabel("Paradigm")
    # Hide unused axes.
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(f"Runtime heatmap ({runtime}, {time_unit})", fontsize=14)
    fig.tight_layout()
    if save_path is not None:
        _save(fig, save_path, "heatmap")
    return fig


CHART_DISPATCH = {
    "line": plot_linechart,
    "bar": plot_barchart,
    "heatmap": plot_heatmap,
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the benchmarking targets and consolidate their Google-Benchmark "
            "JSON reports into a single tidy CSV (plus optional plots)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    discovery = parser.add_argument_group("discovery")
    discovery.add_argument(
        "-p",
        "--path",
        nargs="+",
        type=Path,
        default=[Path(".")],
        help="Directories to search recursively (relative to --build-dir).",
    )
    discovery.add_argument(
        "-r",
        "--regex",
        nargs="+",
        type=str,
        required=True,
        help="Regex pattern(s) matched against file paths to select "
        "executables (or reports with --skip-benchmark).",
    )
    discovery.add_argument(
        "-x",
        "--exclude",
        nargs="+",
        type=str,
        default=[],
        help="Regex pattern(s) to exclude matched paths (e.g. '.*_cpp').",
    )
    discovery.add_argument(
        "-b",
        "--build-dir",
        type=Path,
        default=Path.cwd(),
        help="Build folder used as the working directory for the whole pipeline.",
    )
    discovery.add_argument(
        "-s",
        "--skip-benchmark",
        action="store_true",
        help="Skip running benchmarks; use --path/--regex to locate reports directly.",
    )
    discovery.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Re-run benchmarks even if a report already exists.",
    )
    discovery.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Only list the executables (or reports with --skip-benchmark) "
        "matched by --path/--regex, then exit without doing anything else.",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "-H",
        "--hardware",
        type=str,
        default="",
        help="Hardware identifier stored in the 'Hardware' column (e.g. RTX5080).",
    )
    default_output = datetime.now().strftime("%Y-%m-%d_%H-%M_Benchmark_Result")
    output.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(default_output),
        help="Output base name for the CSV and plots.",
    )
    output.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (-v: DEBUG, -vv: TRACE).",
    )

    chart = parser.add_argument_group("charts")
    chart.add_argument(
        "-c",
        "--chart",
        nargs="+",
        default=[],
        choices=["none", "all", "line", "bar", "heatmap"],
        help="Chart type(s) to generate.",
    )
    chart.add_argument(
        "--chart-time-unit",
        type=str,
        default="ms",
        choices=list(TIME_UNIT_TO_NS),
        help="Time unit for the plots (values are converted accordingly).",
    )
    chart.add_argument(
        "--chart-time-choice",
        type=str,
        default="wall_time",
        choices=list(RUNTIME_CHOICES),
        help="Which runtime to plot.",
    )
    chart.add_argument(
        "--chart-merge-descriptor",
        action="store_true",
        help="Merge Paradigm and Description into a single series for the plots.",
    )
    return parser


def resolve_charts(chart_args: list[str]) -> list[str]:
    """Expand the ``--chart`` selection into a concrete, de-duplicated list."""
    if not chart_args or "none" in chart_args:
        return []
    if "all" in chart_args:
        return list(CHART_DISPATCH)
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(chart_args))


def main() -> int:
    args = build_parser().parse_args()

    logger.remove()
    logger.add(sys.stdout, level=["INFO", "DEBUG", "TRACE"][min(args.verbose, 2)])

    # The build folder is always the working directory.
    build_dir = args.build_dir.resolve()
    if not build_dir.is_dir():
        logger.error(f"Build directory {build_dir} does not exist.")
        return 1
    os.chdir(build_dir)
    logger.info(f"Working directory: {build_dir}")

    # 1. Discover --------------------------------------------------------------
    files = find_files(
        args.path,
        args.regex,
        require_executable=not args.skip_benchmark,
        exclude=args.exclude,
    )

    if args.dry_run:
        print_found_files(files, build_dir, are_reports=args.skip_benchmark)
        return 0 if files else 1

    if not files:
        logger.error("Nothing matched the given path/regex.")
        return 1

    # 2. Benchmark (unless skipped) -------------------------------------------
    if args.skip_benchmark:
        reports = files
    else:
        reports = run_benchmarks(files, force=args.force)
    if not reports:
        logger.error("No report files available to process.")
        return 1

    # 3. Consolidate -----------------------------------------------------------
    df = load_reports(reports, hardware=args.hardware)
    if df.empty:
        return 1
    csv_path = args.output.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    logger.success(f"Wrote unified CSV: {csv_path.resolve()}")

    # 4. Plot ------------------------------------------------------------------
    charts = resolve_charts(args.chart)
    if charts:
        sns.set_theme(style="whitegrid", context="talk")
    for chart in charts:
        logger.info(f"Generating {chart} chart")
        CHART_DISPATCH[chart](
            df,
            args.output,
            args.chart_time_unit,
            args.chart_time_choice,
            args.chart_merge_descriptor,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
