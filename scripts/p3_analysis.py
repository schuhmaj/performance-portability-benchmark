#!/usr/bin/env python3
"""Create Cascade Plots and Navcharts from benchmark result CSV files.

The script consumes one or more tidy CSV files produced by ``benchmark.py``.
Without ``--complexity`` it creates a Cascade Plot of application efficiency
and performance portability.  Supplying ``--complexity`` creates a Navchart
of performance portability and the selected code-complexity metric instead.

Application efficiency is calculated from wall-clock time: for every hardware
and workload, the fastest implementation has efficiency 1 and every other
implementation has ``fastest_runtime / runtime``.  Efficiencies are averaged
over the selected workloads, then performance portability is calculated as
their harmonic mean over the complete set of selected hardware.  Missing
hardware/workload results therefore give an implementation a PP score of zero.
"""

from __future__ import annotations

import argparse
import re
import sys
from itertools import cycle
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

BENCHMARK_PROBLEM = "Benchmark Problem"
PARADIGM = "Paradigm"
DESCRIPTION = "Description"
PRECISION = "Precision"
HARDWARE = "Hardware"
PROBLEM_SIZE = "Problem Size"
WALL_CLOCK_TIME = "Wall Clock Time"
TIME_UNIT = "Time Unit"

APPLICATION = "Application"
APPLICATION_EFFICIENCY = "Application Efficiency"
PERFORMANCE_PORTABILITY = "Performance Portability"

COMPLEXITY_NAME = "Name"
COMPLEXITY_FRAMEWORK = "Framework"

TIME_UNIT_TO_NS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}

METRIC_ALIASES = {
    "sloc": "SLOC",
    "halstead-volume": "Halstead Volume",
    "volume": "Halstead Volume",
    "v": "Halstead Volume",
    "halstead-difficulty": "Halstead Difficulty",
    "difficulty": "Halstead Difficulty",
    "d": "Halstead Difficulty",
    "halstead-effort": "Halstead Effort",
    "effort": "Halstead Effort",
    "e": "Halstead Effort",
}

MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "p")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create a Cascade Plot, or a Navchart when --complexity is given, "
            "from benchmark.py CSV output."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        metavar="CSV",
        help="One or more CSV files produced by benchmark.py.",
    )
    parser.add_argument(
        "-n",
        "--name",
        required=True,
        help=(
            "Benchmark problem to plot. An exact case-insensitive match is "
            "preferred; otherwise a unique substring match is accepted."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (-v: DEBUG, -vv: TRACE).",
    )
    parser.add_argument(
        "-f",
        "--filter",
        dest="description_filter",
        help="Regular expression matched against the Description column.",
    )
    parser.add_argument(
        "-c",
        "--complexity",
        type=Path,
        help="Code-complexity CSV. Supplying it selects a Navchart.",
    )
    parser.add_argument(
        "--complexity-metric",
        default="halstead-effort",
        help=(
            "Navchart metric: SLOC, Halstead volume, Halstead difficulty, or "
            "Halstead effort (common short aliases are accepted)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output plot path. If no suffix is provided, .pdf is appended. "
            "Defaults to <problem>_cascade.pdf or <problem>_navchart.pdf."
        ),
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize complexity to the CPP implementation (CPP = 100%%).",
    )
    return parser


def configure_logging(verbosity: int) -> None:
    """Configure Loguru consistently with ``benchmark.py``.

    Args:
        verbosity: Number of ``-v`` flags supplied by the user.
    """
    logger.remove()
    logger.add(sys.stdout, level=["INFO", "DEBUG", "TRACE"][min(verbosity, 2)])


def _require_columns(df: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    """Validate that a DataFrame contains a set of columns.

    Args:
        df: DataFrame to validate.
        columns: Required column names.
        source: Input file used in an error message.

    Raises:
        ValueError: If one or more columns are absent.
    """
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def load_benchmark_csvs(paths: list[Path]) -> pd.DataFrame:
    """Read and concatenate benchmark result CSV files.

    Args:
        paths: CSV files produced by ``benchmark.py``.

    Returns:
        Concatenated benchmark data with a source-file column.

    Raises:
        FileNotFoundError: If an input path does not exist.
        ValueError: If a file does not have the benchmark output schema.
    """
    required = {
        BENCHMARK_PROBLEM,
        PARADIGM,
        DESCRIPTION,
        HARDWARE,
        PROBLEM_SIZE,
        WALL_CLOCK_TIME,
        TIME_UNIT,
    }
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark CSV does not exist: {path}")
        frame = pd.read_csv(path)
        _require_columns(frame, required, path)
        frame["Source File"] = str(path)
        frames.append(frame)
        logger.debug(f"Read {len(frame)} rows from {path}")

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(combined)} benchmark rows from {len(paths)} file(s)")
    return combined


def _resolve_unique_value(values: pd.Series, query: str, label: str) -> str:
    """Resolve a user query to one exact or uniquely partial-matching value.

    Args:
        values: Candidate values.
        query: User-supplied query.
        label: Human-readable value type for errors.

    Returns:
        The resolved value as it appears in the input.

    Raises:
        ValueError: If no value or more than one value matches.
    """
    candidates = sorted({str(value) for value in values.dropna().unique()})
    exact = [value for value in candidates if value.casefold() == query.casefold()]
    if len(exact) == 1:
        return exact[0]

    partial = [value for value in candidates if query.casefold() in value.casefold()]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise ValueError(
            f"No {label} matches {query!r}. Available values: {candidates}"
        )
    raise ValueError(
        f"{label.capitalize()} {query!r} is ambiguous; matches: {partial}"
    )


def _compile_description_filter(pattern: str | None) -> re.Pattern[str] | None:
    """Compile the optional description regular expression.

    Args:
        pattern: User-supplied regular expression, or ``None``.

    Returns:
        A compiled expression, or ``None``.

    Raises:
        ValueError: If the expression is invalid.
    """
    if pattern is None:
        return None
    try:
        return re.compile(pattern)
    except re.error as error:
        raise ValueError(f"Invalid description regex {pattern!r}: {error}") from error


def _description_is_workload(df: pd.DataFrame) -> bool:
    """Infer whether Description identifies workloads rather than variants.

    A descriptor is treated as a workload dimension when each selected,
    non-empty description is implemented by at least half of the paradigms.
    This identifies the polyhedral model names while retaining implementation
    variants such as ``Cuda[Naive]`` and ``Cuda[Cublas]`` as applications.

    Args:
        df: Filtered benchmark rows.

    Returns:
        Whether Description should be part of the workload identity.
    """
    described = df.loc[df[DESCRIPTION].ne("")]
    if described.empty:
        return False
    paradigm_count = max(df[PARADIGM].nunique(), 1)
    coverage = described.groupby(DESCRIPTION)[PARADIGM].nunique() / paradigm_count
    logger.debug(f"Description coverage ratios: {coverage.to_dict()}")
    return bool((coverage >= 0.5).all())


def select_problem_rows(
    df: pd.DataFrame,
    name: str,
    description_pattern: str | None,
) -> tuple[pd.DataFrame, str, bool]:
    """Select a problem and optionally filter its descriptions.

    Args:
        df: Combined benchmark results.
        name: Requested benchmark problem.
        description_pattern: Optional description regular expression.

    Returns:
        A tuple of filtered rows, resolved problem name, and a flag indicating
        whether Description is a workload dimension.

    Raises:
        ValueError: If selection leaves no rows or required values are invalid.
    """
    resolved_name = _resolve_unique_value(df[BENCHMARK_PROBLEM], name, "problem")
    selected = df.loc[df[BENCHMARK_PROBLEM] == resolved_name].copy()
    selected[PARADIGM] = selected[PARADIGM].fillna("").astype(str).str.strip()
    selected[DESCRIPTION] = selected[DESCRIPTION].fillna("").astype(str).str.strip()
    selected[HARDWARE] = selected[HARDWARE].fillna("").astype(str).str.strip()

    if selected[PARADIGM].eq("").any():
        raise ValueError("Selected benchmark rows contain an empty Paradigm value.")
    if selected[HARDWARE].eq("").any():
        raise ValueError("Selected benchmark rows contain an empty Hardware value.")

    # Classify Description before filtering so a single selected variant does
    # not look like a workload implemented by every remaining paradigm.
    description_is_workload = _description_is_workload(selected)

    expression = _compile_description_filter(description_pattern)
    if expression is not None:
        mask = selected[DESCRIPTION].map(
            lambda value: expression.search(value) is not None
        )
        selected = selected.loc[mask].copy()
        if selected.empty:
            raise ValueError(
                f"Description regex {description_pattern!r} matched no rows for "
                f"{resolved_name}."
            )

    logger.info(
        f"Selected {len(selected)} rows for {resolved_name} on "
        f"{selected[HARDWARE].nunique()} hardware platform(s)"
    )
    logger.debug(
        "Treating Description as a "
        + ("workload dimension" if description_is_workload else "variant label")
    )
    return selected, resolved_name, description_is_workload


def _application_label(paradigm: str, description: str, is_workload: bool) -> str:
    """Construct an application label from a paradigm and description.

    Args:
        paradigm: Programming paradigm or framework.
        description: Optional implementation description.
        is_workload: Whether descriptions identify workloads.

    Returns:
        A display/application identity.
    """
    if description and not is_workload:
        return f"{paradigm}[{description}]"
    return paradigm


def _convert_runtime_to_ns(df: pd.DataFrame) -> pd.Series:
    """Convert wall-clock runtimes to nanoseconds.

    Args:
        df: Selected benchmark rows.

    Returns:
        Numeric runtimes expressed in nanoseconds.

    Raises:
        ValueError: If a time unit is unsupported.
    """
    units = df[TIME_UNIT].fillna("").astype(str).str.strip().str.lower()
    unknown = sorted(set(units) - set(TIME_UNIT_TO_NS))
    if unknown:
        raise ValueError(
            f"Unsupported values in {TIME_UNIT!r}: {unknown}. "
            f"Supported units: {sorted(TIME_UNIT_TO_NS)}"
        )
    runtimes = pd.to_numeric(df[WALL_CLOCK_TIME], errors="coerce")
    factors = units.map(TIME_UNIT_TO_NS).astype(float)
    return runtimes * factors


def calculate_metrics(
    df: pd.DataFrame,
    description_is_workload: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate application efficiency and performance portability.

    Duplicate measurements are reduced to their median runtime. Application
    efficiencies are computed per hardware/workload and then arithmetically
    averaged over workloads. PP is the harmonic mean across all selected
    hardware; an absent result contributes zero and makes PP zero.

    Args:
        df: Selected benchmark rows.
        description_is_workload: Whether Description belongs to the workload
            key rather than the application label.

    Returns:
        A pair ``(efficiency, portability)``. The first DataFrame has one row
        per application/hardware; the second has one row per application.

    Raises:
        ValueError: If there are no positive numeric wall-clock runtimes.
    """
    data = df.copy()
    data[APPLICATION] = [
        _application_label(paradigm, description, description_is_workload)
        for paradigm, description in zip(data[PARADIGM], data[DESCRIPTION])
    ]
    data["Runtime (ns)"] = _convert_runtime_to_ns(data)
    invalid = data["Runtime (ns)"].isna() | data["Runtime (ns)"].le(0)
    if invalid.any():
        logger.warning(f"Dropping {int(invalid.sum())} rows with invalid runtimes")
        data = data.loc[~invalid].copy()
    if data.empty:
        raise ValueError("No positive numeric wall-clock runtimes remain.")

    workload_columns = [PROBLEM_SIZE]
    if PRECISION in data.columns and data[PRECISION].notna().any():
        workload_columns.append(PRECISION)
    if description_is_workload:
        workload_columns.append(DESCRIPTION)

    measurement_keys = [HARDWARE, APPLICATION, *workload_columns]
    runtimes = (
        data.groupby(measurement_keys, dropna=False, as_index=False)["Runtime (ns)"]
        .median()
        .sort_values(measurement_keys)
    )
    best_keys = [HARDWARE, *workload_columns]
    runtimes["Best Runtime (ns)"] = runtimes.groupby(best_keys, dropna=False)[
        "Runtime (ns)"
    ].transform("min")
    runtimes[APPLICATION_EFFICIENCY] = (
        runtimes["Best Runtime (ns)"] / runtimes["Runtime (ns)"]
    )

    applications = sorted(runtimes[APPLICATION].unique())
    hardware = sorted(runtimes[HARDWARE].unique())
    workloads = runtimes[workload_columns].drop_duplicates()
    full_index = pd.MultiIndex.from_frame(
        pd.DataFrame(
            [
                (application, platform, *workload)
                for application in applications
                for platform in hardware
                for workload in workloads.itertuples(index=False, name=None)
            ],
            columns=[APPLICATION, HARDWARE, *workload_columns],
        )
    )
    indexed = runtimes.set_index([APPLICATION, HARDWARE, *workload_columns])
    complete = indexed.reindex(full_index)
    complete.index.names = [APPLICATION, HARDWARE, *workload_columns]
    complete[APPLICATION_EFFICIENCY] = complete[APPLICATION_EFFICIENCY].fillna(0.0)
    complete = complete.reset_index()

    efficiency = (
        complete.groupby([APPLICATION, HARDWARE], as_index=False)[
            APPLICATION_EFFICIENCY
        ]
        .mean()
        .sort_values([APPLICATION, HARDWARE])
    )

    def harmonic_mean_or_zero(values: pd.Series) -> float:
        """Calculate a harmonic mean, returning zero for unsupported values."""
        numeric = values.to_numpy(dtype=float)
        if len(numeric) != len(hardware) or np.any(numeric <= 0.0):
            return 0.0
        return float(len(numeric) / np.reciprocal(numeric).sum())

    portability = (
        efficiency.groupby(APPLICATION, as_index=False)[APPLICATION_EFFICIENCY]
        .agg(harmonic_mean_or_zero)
        .rename(columns={APPLICATION_EFFICIENCY: PERFORMANCE_PORTABILITY})
        .sort_values(PERFORMANCE_PORTABILITY, ascending=False)
    )
    logger.debug(f"Performance portability:\n{portability.to_string(index=False)}")
    return efficiency, portability


def _slugify(value: str) -> str:
    """Convert a value into a compact filename component.

    Args:
        value: Arbitrary text.

    Returns:
        A filesystem-friendly lowercase slug.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "plot"


def resolve_output_path(output: Path | None, problem: str, navchart: bool) -> Path:
    """Resolve the plot output path and default PDF extension.

    Args:
        output: User-supplied path, or ``None``.
        problem: Resolved problem name.
        navchart: Whether the output is a Navchart.

    Returns:
        Output path with a suffix.
    """
    kind = "navchart" if navchart else "cascade"
    path = output or Path(f"{_slugify(problem)}_{kind}.pdf")
    if not path.suffix:
        path = path.with_suffix(".pdf")
    return path


def _plot_styles(names: list[str]) -> tuple[dict[str, object], dict[str, str]]:
    """Assign stable colors and markers to application names.

    Args:
        names: Ordered application names.

    Returns:
        Color and marker mappings keyed by application.
    """
    colors = sns.color_palette("tab20", n_colors=max(len(names), 1))
    marker_cycle = cycle(MARKERS)
    return dict(zip(names, colors)), {name: next(marker_cycle) for name in names}


def plot_cascade(
    efficiency: pd.DataFrame,
    portability: pd.DataFrame,
    problem: str,
) -> plt.Figure:
    """Create a Cascade Plot following the P3 Analysis Library layout.

    Args:
        efficiency: Application efficiency by application and hardware.
        portability: Performance portability by application.
        problem: Problem name used in the title.

    Returns:
        The Matplotlib figure.
    """
    applications = portability[APPLICATION].tolist()
    platforms = sorted(efficiency[HARDWARE].unique())
    colors, markers = _plot_styles(applications)
    platform_colors = dict(
        zip(platforms, sns.color_palette("vlag", n_colors=max(len(platforms), 1)))
    )
    platform_labels = {
        platform: chr(ord("A") + index) if index < 26 else str(index + 1)
        for index, platform in enumerate(platforms)
    }

    figure_width = max(8.0, 6.0 + 0.22 * len(applications))
    figure_height = max(6.0, 4.8 + 0.25 * len(applications))
    figure = plt.figure(figsize=(figure_width, figure_height))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[4.5, max(1.5, 0.34 * len(applications))],
        width_ratios=[5.5, max(1.5, 0.2 * len(applications))],
        hspace=0.03,
        wspace=0.05,
    )
    efficiency_axis = figure.add_subplot(grid[0, 0])
    platform_axis = figure.add_subplot(grid[1, 0], sharex=efficiency_axis)
    portability_axis = figure.add_subplot(grid[0, 1], sharey=efficiency_axis)

    for application in applications:
        rows = efficiency.loc[
            (efficiency[APPLICATION] == application)
            & efficiency[APPLICATION_EFFICIENCY].gt(0.0)
        ].sort_values(APPLICATION_EFFICIENCY, ascending=False)
        ranks = np.arange(1, len(rows) + 1)
        efficiency_axis.plot(
            ranks,
            rows[APPLICATION_EFFICIENCY],
            color=colors[application],
            marker=markers[application],
            linewidth=1.6,
            markersize=7,
            label=application,
        )

    efficiency_axis.set_ylabel("Application Efficiency")
    efficiency_axis.set_ylim(0.0, 1.05)
    efficiency_axis.set_xlim(0.5, len(platforms) + 0.5)
    efficiency_axis.set_xticks(np.arange(1, len(platforms) + 1))
    efficiency_axis.tick_params(axis="x", labelbottom=False)
    efficiency_axis.grid(True, linestyle="--", alpha=0.45)
    figure.suptitle(f"{problem} Cascade Plot", y=0.98)

    row_height = 1.0
    for row_index, application in enumerate(reversed(applications)):
        rows = efficiency.loc[
            (efficiency[APPLICATION] == application)
            & efficiency[APPLICATION_EFFICIENCY].gt(0.0)
        ].sort_values(APPLICATION_EFFICIENCY, ascending=False)
        y = row_index * row_height
        platform_axis.plot(
            [0.5, len(platforms) + 0.5],
            [y + 0.5, y + 0.5],
            color=colors[application],
            marker=markers[application],
            markersize=6,
            linewidth=1.2,
            zorder=1,
        )
        for rank, platform in enumerate(rows[HARDWARE], start=1):
            platform_axis.add_patch(
                Rectangle(
                    (rank - 0.5, y),
                    1.0,
                    row_height,
                    facecolor=platform_colors[platform],
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=2,
                )
            )
            platform_axis.text(
                rank,
                y + 0.5,
                platform_labels[platform],
                ha="center",
                va="center",
                fontsize=8,
                zorder=3,
            )
    platform_axis.set_xlabel("Platform rank (highest efficiency first)")
    platform_axis.set_ylim(0.0, max(len(applications), 1))
    platform_axis.set_yticks([])
    platform_axis.grid(False)

    ordered_pp = portability.set_index(APPLICATION).reindex(applications)
    portability_axis.bar(
        np.arange(len(applications)),
        ordered_pp[PERFORMANCE_PORTABILITY],
        color="white",
        edgecolor=[colors[name] for name in applications],
        linewidth=1.5,
    )
    portability_axis.scatter(
        np.arange(len(applications)),
        ordered_pp[PERFORMANCE_PORTABILITY],
        color=[colors[name] for name in applications],
        marker="o",
        zorder=3,
    )
    portability_axis.set_xticks([])
    portability_axis.set_title("Performance\nPortability", fontsize=11)
    portability_axis.yaxis.tick_right()
    portability_axis.grid(True, axis="y", linestyle="--", alpha=0.45)

    application_handles = [
        Line2D(
            [0],
            [0],
            color=colors[name],
            marker=markers[name],
            linewidth=1.5,
            label=name,
        )
        for name in applications
    ]
    platform_handles = [
        Patch(
            facecolor=platform_colors[name],
            edgecolor="black",
            label=f"{platform_labels[name]}: {name}",
        )
        for name in platforms
    ]
    app_legend = figure.legend(
        handles=application_handles,
        title="Application",
        loc="center left",
        bbox_to_anchor=(0.79, 0.55),
        frameon=True,
    )
    figure.add_artist(app_legend)
    figure.legend(
        handles=platform_handles,
        title="Platform",
        loc="lower center",
        bbox_to_anchor=(0.43, 0.01),
        ncol=min(4, max(len(platforms), 1)),
        frameon=True,
    )
    figure.subplots_adjust(right=0.76, bottom=0.25, top=0.9)
    return figure


def _normalize_token(value: str) -> str:
    """Normalize a framework label for case-insensitive matching.

    Args:
        value: Framework or application label.

    Returns:
        Lowercase alphanumeric token.
    """
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _calculate_halstead_columns(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Add derived Halstead metrics to a complexity DataFrame.

    Args:
        df: Complexity data containing n1, n2, N1, and N2.
        source: Input file used in error messages.

    Returns:
        Copy of the data with derived metric columns.

    Raises:
        ValueError: If the primitive Halstead columns are absent or invalid.
    """
    required = ["n1", "n2", "N1", "N2"]
    _require_columns(df, required, source)
    result = df.copy()
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[required].isna().any(axis=None):
        raise ValueError(f"{source} contains non-numeric Halstead counts.")

    vocabulary = result["n1"] + result["n2"]
    length = result["N1"] + result["N2"]
    result["Halstead Volume"] = np.where(
        vocabulary > 0.0, length * np.log2(vocabulary), np.nan
    )
    result["Halstead Difficulty"] = np.where(
        result["n2"] > 0.0,
        (result["n1"] / 2.0) * (result["N2"] / result["n2"]),
        np.nan,
    )
    result["Halstead Effort"] = (
        result["Halstead Volume"] * result["Halstead Difficulty"]
    )
    return result


def resolve_complexity_metric(
    df: pd.DataFrame,
    requested: str,
    source: Path,
) -> tuple[pd.DataFrame, str]:
    """Resolve or calculate a requested code-complexity metric.

    Args:
        df: Raw complexity data.
        requested: Metric name or alias supplied on the command line.
        source: Complexity CSV path.

    Returns:
        A pair containing enriched complexity data and the canonical metric
        column name.

    Raises:
        ValueError: If the metric is unknown or cannot be calculated.
    """
    alias = requested.casefold().replace("_", "-").replace(" ", "-")
    canonical = METRIC_ALIASES.get(alias)
    if canonical is None:
        direct = [
            column
            for column in df.columns
            if column.casefold() == requested.casefold()
        ]
        if len(direct) != 1:
            raise ValueError(
                f"Unknown complexity metric {requested!r}. Use one of: "
                "SLOC, Halstead volume, Halstead difficulty, Halstead effort."
            )
        canonical = direct[0]

    enriched = df.copy()
    if canonical.startswith("Halstead") and canonical not in enriched.columns:
        enriched = _calculate_halstead_columns(enriched, source)
    if canonical not in enriched.columns:
        raise ValueError(f"{source} does not contain metric column {canonical!r}.")
    enriched[canonical] = pd.to_numeric(enriched[canonical], errors="coerce")
    if enriched[canonical].isna().all():
        raise ValueError(f"Complexity metric {canonical!r} has no numeric values.")
    return enriched, canonical


def load_complexity_data(
    path: Path,
    problem_query: str,
    metric_request: str,
    normalize: bool,
) -> tuple[pd.DataFrame, str]:
    """Load, select, calculate, and optionally normalize complexity data.

    Args:
        path: Complexity CSV path.
        problem_query: User-supplied benchmark problem query.
        metric_request: Requested metric or alias.
        normalize: Whether to express values relative to CPP.

    Returns:
        A pair of selected complexity rows and the plotted metric label.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the input is invalid or no CPP baseline is available.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Complexity CSV does not exist: {path}")
    raw = pd.read_csv(path)
    _require_columns(raw, [COMPLEXITY_NAME, COMPLEXITY_FRAMEWORK], path)
    problem = _resolve_unique_value(
        raw[COMPLEXITY_NAME], problem_query, "complexity problem"
    )
    selected = raw.loc[raw[COMPLEXITY_NAME] == problem].copy()
    selected[COMPLEXITY_FRAMEWORK] = (
        selected[COMPLEXITY_FRAMEWORK].fillna("").astype(str).str.strip()
    )
    selected, metric = resolve_complexity_metric(selected, metric_request, path)
    selected = (
        selected.groupby(COMPLEXITY_FRAMEWORK, as_index=False)[metric]
        .median()
        .dropna(subset=[metric])
    )

    label = metric
    if normalize:
        tokens = selected[COMPLEXITY_FRAMEWORK].map(_normalize_token)
        baseline_rows = selected.loc[tokens.isin({"cpp", "cplusplus", "cpu"}), metric]
        if len(baseline_rows) != 1 or baseline_rows.iloc[0] <= 0.0:
            raise ValueError(
                "--normalize requires exactly one positive CPP complexity row "
                f"for {problem}; found {len(baseline_rows)}."
            )
        baseline = float(baseline_rows.iloc[0])
        selected[metric] = selected[metric] / baseline * 100.0
        label = f"{metric} (% of CPP)"
        selected = selected.rename(columns={metric: label})
        logger.debug(f"Normalized {metric} to CPP baseline {baseline:g}")
    logger.info(f"Loaded {len(selected)} complexity rows for {problem}")
    return selected, label


def _complexity_candidates(application: str) -> list[str]:
    """Generate framework labels that may match an application.

    Args:
        application: Application label, possibly ``Paradigm[Description]``.

    Returns:
        Candidate labels ordered from most to least specific.
    """
    candidates = [application]
    match = re.fullmatch(r"(.+?)\[(.+)]", application)
    if match:
        paradigm, description = match.groups()
        candidates.extend([description, paradigm])
    return list(dict.fromkeys(candidates))


def merge_portability_complexity(
    portability: pd.DataFrame,
    complexity: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Match application PP values with code-complexity rows.

    Exact application labels are preferred. For labels containing a
    description, the description alone and then the base paradigm are tried as
    fallbacks; this maps ``Cuda[Cublas]`` to ``Cublas`` and polyhedral model
    labels such as ``Cuda[Eros]`` to ``Cuda``.

    Args:
        portability: Performance portability per application.
        complexity: Complexity per framework.
        metric: Complexity metric column.

    Returns:
        Matched application, PP, and complexity values.

    Raises:
        ValueError: If none of the applications can be matched.
    """
    lookup = {
        _normalize_token(row[COMPLEXITY_FRAMEWORK]): float(row[metric])
        for _, row in complexity.iterrows()
    }
    rows: list[dict[str, object]] = []
    unmatched: list[str] = []
    for _, row in portability.iterrows():
        application = str(row[APPLICATION])
        value = None
        for candidate in _complexity_candidates(application):
            token = _normalize_token(candidate)
            if token in lookup:
                value = lookup[token]
                break
        if value is None:
            unmatched.append(application)
            continue
        rows.append(
            {
                APPLICATION: application,
                PERFORMANCE_PORTABILITY: float(row[PERFORMANCE_PORTABILITY]),
                metric: value,
            }
        )
    if unmatched:
        logger.warning(
            "No complexity value for application(s), omitting them: "
            + ", ".join(unmatched)
        )
    if not rows:
        raise ValueError("No application labels match the complexity Framework values.")
    return pd.DataFrame(rows)


def plot_navchart(data: pd.DataFrame, metric: str, problem: str) -> plt.Figure:
    """Create a Navchart of complexity and performance portability.

    Args:
        data: Matched application PP and complexity data.
        metric: Complexity metric column and x-axis label.
        problem: Problem name used in the title.

    Returns:
        The Matplotlib figure.
    """
    applications = data[APPLICATION].tolist()
    colors, markers = _plot_styles(applications)
    figure, axis = plt.subplots(figsize=(7.5, 6.0))
    for _, row in data.iterrows():
        application = str(row[APPLICATION])
        axis.scatter(
            row[metric],
            row[PERFORMANCE_PORTABILITY],
            color=colors[application],
            marker=markers[application],
            s=95,
            edgecolor="black",
            linewidth=0.5,
            label=application,
            zorder=3,
        )

    axis.set_xlabel(metric)
    axis.set_ylabel("Performance Portability")
    axis.set_ylim(0.0, 1.05)
    axis.set_title(f"{problem} Navigation Chart")
    axis.grid(True, linestyle="--", alpha=0.45)
    axis.legend(
        title="Application",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )
    figure.tight_layout()
    return figure


def save_figure(figure: plt.Figure, output: Path) -> None:
    """Save a figure, creating its parent directory if necessary.

    Args:
        figure: Figure to save.
        output: Destination path.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    logger.success(f"Wrote plot: {output.resolve()}")


def main() -> int:
    """Run the P3 analysis command-line program.

    Returns:
        Process exit status: zero on success, one on failure.
    """
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    sns.set_theme(style="whitegrid", context="talk")

    try:
        benchmark_data = load_benchmark_csvs(args.csv_files)
        selected, problem, description_is_workload = select_problem_rows(
            benchmark_data, args.name, args.description_filter
        )
        efficiency, portability = calculate_metrics(
            selected, description_is_workload
        )
        output = resolve_output_path(
            args.output, problem, navchart=args.complexity is not None
        )

        if args.complexity is None:
            if args.normalize:
                raise ValueError("--normalize is only valid with --complexity.")
            figure = plot_cascade(efficiency, portability, problem)
        else:
            complexity, metric = load_complexity_data(
                args.complexity,
                args.name,
                args.complexity_metric,
                args.normalize,
            )
            navchart_data = merge_portability_complexity(
                portability, complexity, metric
            )
            logger.debug(f"Navchart data:\n{navchart_data.to_string(index=False)}")
            figure = plot_navchart(navchart_data, metric, problem)

        save_figure(figure, output)
        return 0
    except (FileNotFoundError, OSError, ValueError) as error:
        logger.error(str(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
