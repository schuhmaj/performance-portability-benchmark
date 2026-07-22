#!/usr/bin/env python3
"""Create Cascade Plots and Navcharts from benchmark result CSV files.

The script consumes one or more tidy CSV files produced by ``benchmark.py``.
The ``--chart`` option explicitly selects a Cascade Plot, a Navchart of
performance portability versus code complexity, or a combined plot.  The
combined plot includes the Cascade and Navchart panels plus performance-
portability scaling over problem size. The benchmark problem is supplied as
the first positional argument.

Application efficiency is calculated from wall-clock time: for every hardware
and workload, the fastest implementation has efficiency 1 and every other
implementation has ``fastest_runtime / runtime``.  Efficiencies are averaged
over the selected workloads, then performance portability is calculated as
their harmonic mean over the complete set of selected hardware.  Missing
hardware/workload results therefore give an implementation a PP score of zero.
With ``--non-zero-pp``, unsupported platforms are excluded from the PP
harmonic mean instead, without changing the application-efficiency data.
With ``--export-to-csv``, application-efficiency and performance-portability
data are written separately for every problem-size/precision combination,
along with per-precision averages over problem size.

References:
    S.J. Pennycook, J.D. Sewall, D. Jacobsen, T. Deakin and S. McIntosh-Smith, "Navigating Performance Portability", in Computing in Science & Engineering, Volume: 23, Issue: 5, 01 Sept.-Oct. 2021

See Also:
    https://github.com/P3HPC/p3-analysis-library

"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import EngFormatter

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
PROBLEM = "Problem"
# Matplotlib's math text uses two negative thin spaces to approximate the
# ``\kern-0.3em`` overlap in the conventional reflected-P PP symbol.
PP_SYMBOL = r"$\mathrm{ꟼ\!\!P}$"

COMPLEXITY_NAME = "Name"
COMPLEXITY_FRAMEWORK = "Framework"
AVERAGE_SIZE = "average"
BEST_SIZE = "best"
WORST_SIZE = "worst"

TIME_UNIT_TO_NS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}

METRIC_ALIASES = {
    "sloc": "SLOC",
    "halstead-vocabulary": "Halstead Vocabulary",
    "vocabulary": "Halstead Vocabulary",
    "Eta": "Halstead Vocabulary",
    "halstead-program-length": "Halstead Program Length",
    "halstead-length": "Halstead Program Length",
    "program-length": "Halstead Program Length",
    "length": "Halstead Program Length",
    "N": "Halstead Program Length",
    "halstead-volume": "Halstead Volume",
    "volume": "Halstead Volume",
    "V": "Halstead Volume",
    "halstead-difficulty": "Halstead Difficulty",
    "difficulty": "Halstead Difficulty",
    "D": "Halstead Difficulty",
    "halstead-effort": "Halstead Effort",
    "effort": "Halstead Effort",
    "E": "Halstead Effort",
}

PLOT_MARKER_SIZE = 10
LEGEND_MARKER_SIZE = 11
SCATTER_MARKER_AREA = 140

TAB20 = matplotlib.colormaps["tab20"].colors
FRAMEWORK_COLOR_MAP = {
    "CPP": TAB20[0],
    "Stdpar": TAB20[1],
    "Vulkan": TAB20[2],
    "Slang-Vulkan": TAB20[3],
    "AdaptiveCpp": TAB20[4],
    "Kokkos": TAB20[6],
    "RAJA": TAB20[10],
    "Alpaka": TAB20[12],
    "Cuda": TAB20[8],
    "Slang-Cuda": TAB20[9],
    "Cublas": TAB20[13],
    "Hip": "black",
    "OpenACC": TAB20[14],
    "OpenMP": TAB20[18],
    "OpenCL": TAB20[16],
    "Boost": TAB20[17],
}

HARDWARE_VENDOR_PATTERNS = {
    "NVIDIA": re.compile(
        r"(?i)\b(nvidia|geforce|quadro|tesla|rtx\d*|gtx\d*|gh\d+|h\d{2,3}|a\d{2,3})\b"
    ),
    "AMD": re.compile(r"(?i)\b(amd|radeon|instinct|mi\d+)\b"),
    "Intel": re.compile(r"(?i)\b(intel|xeon|arc|pvc)\b"),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create a Cascade Plot, Navchart, or combined P3 plot from "
            "benchmark.py CSV output."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    general = parser.add_argument_group("general script options")
    general.add_argument(
        "name",
        metavar="NAME",
        help=(
            "Benchmark problem to plot. Exact case-insensitive matches are "
            "preferred; a unique substring match is accepted."
        ),
    )
    general.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        metavar="CSV",
        help="One or more CSV files produced by benchmark.py.",
    )
    general.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (-v: DEBUG, -vv: TRACE).",
    )
    general.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output plot path. If no suffix is provided, .pdf is appended. "
            "Defaults to <problem>_<cascade|navchart|combined>.pdf."
        ),
    )
    general.add_argument(
        "-e",
        "--export-to-csv",
        action="store_true",
        help=(
            "Export application-efficiency and performance-portability data "
            "to <plot-prefix>_application_efficiency.csv and "
            "<plot-prefix>_performance_portability.csv. Exported metrics "
            "include separate and average rows for all sizes and precisions, "
            "and all available complexity metrics when --complexity is "
            "supplied."
        ),
    )
    general.add_argument(
        "-c",
        "--chart",
        choices=("cascade", "navchart", "combined"),
        default="cascade",
        help="Chart to create. Navchart and combined require --complexity.",
    )
    general.add_argument(
        "-l",
        "--legend",
        action="store_true",
        help=(
            "Omit legends from the plot and save them as a separate PDF with "
            "four columns."
        ),
    )
    general.add_argument(
        "--remove-description",
        action="store_true",
        help="Remove bracketed descriptions such as [Naive] from legends.",
    )

    complexity = parser.add_argument_group("code complexity options")
    metrics = parser.add_argument_group(
        "performance portability / application efficiency metric options"
    )
    metrics.add_argument(
        "-i",
        "--include",
        dest="description_include",
        help=("Only keep rows whose Description matches this regular expression."),
    )
    metrics.add_argument(
        "-x",
        "--exclude",
        dest="description_exclude",
        help="Exclude rows whose Description matches this regular expression.",
    )
    metrics.add_argument(
        "-s",
        "--size",
        type=parse_problem_size,
        help=(
            "Use this exact Problem Size for application efficiency and the "
            "aggregate PP/complexity point; 'avg', 'average', or 'mean' takes "
            "the arithmetic mean; 'best' takes the maximum; and 'worst' "
            "takes the minimum of each metric over problem sizes. Scaling "
            "still uses all sizes."
        ),
    )
    metrics.add_argument(
        "-p",
        "--precision",
        type=int,
        choices=[32, 64],
        help="Keep results with this floating-point precision.",
    )
    metrics.add_argument(
        "--non-zero-pp",
        action="store_true",
        help=(
            "Calculate PP over supported platforms only. Missing platforms "
            "remain zero in application-efficiency plots."
        ),
    )
    metrics.add_argument(
        "--log-size",
        action="store_true",
        help="Use a logarithmic problem-size axis in the combined scaling plot.",
    )

    complexity.add_argument(
        "--complexity",
        type=Path,
        help=(
            "Code-complexity CSV used by navchart and combined charts, and "
            "included in CSV exports."
        ),
    )
    complexity.add_argument(
        "--complexity-metric",
        default="halstead-effort",
        help=(
            "Navchart metric: SLOC, Halstead vocabulary, Halstead program "
            "length, Halstead volume, Halstead difficulty, or Halstead effort "
            "(common short aliases are accepted)."
        ),
    )
    complexity_scaling = complexity.add_mutually_exclusive_group()
    complexity_scaling.add_argument(
        "--normalize",
        action="store_true",
        help="Divide complexity by the CPP score (CPP = 100%%).",
    )
    complexity_scaling.add_argument(
        "--additive",
        action="store_true",
        help="Subtract the CPP complexity score from every paradigm score.",
    )
    complexity.add_argument(
        "--log-complexity",
        action="store_true",
        help="Use a logarithmic complexity axis in navchart/combined plots.",
    )
    return parser


def parse_problem_size(value: str) -> float | str:
    """Parse a numeric problem size or a size-summary sentinel."""
    normalized = value.casefold()
    if normalized in {"avg", AVERAGE_SIZE, "mean"}:
        return AVERAGE_SIZE
    if normalized in {BEST_SIZE, WORST_SIZE}:
        return normalized
    try:
        return float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "problem size must be numeric, 'avg', 'average', 'mean', "
            "'best', or 'worst'"
        ) from error


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
        PRECISION,
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
    raise ValueError(f"{label.capitalize()} {query!r} is ambiguous; matches: {partial}")


def _compile_description_filter(
    pattern: str | None, option: str
) -> re.Pattern[str] | None:
    """Compile an optional description regular expression.

    Args:
        pattern: User-supplied regular expression, or ``None``.
        option: Command-line option name used in errors.

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
        raise ValueError(
            f"Invalid {option} description regex {pattern!r}: {error}"
        ) from error


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


def _filter_problem_size(
    df: pd.DataFrame, problem_size: float, problem_name: str
) -> pd.DataFrame:
    """Select one exact numeric problem size from already filtered rows."""
    numeric_sizes = pd.to_numeric(df[PROBLEM_SIZE], errors="coerce")
    size_mask = np.isclose(
        numeric_sizes.to_numpy(dtype=float), problem_size, equal_nan=False
    )
    selected = df.loc[size_mask].copy()
    if selected.empty:
        available = sorted(numeric_sizes.dropna().unique())
        raise ValueError(
            f"Problem size {problem_size:g} has no rows for {problem_name}. "
            f"Available sizes: {available}"
        )
    return selected


def select_problem_rows(
    df: pd.DataFrame,
    name: str,
    description_include: str | None,
    description_exclude: str | None,
    problem_size: float | None,
    precision: int | None,
) -> tuple[pd.DataFrame, str, bool]:
    """Select a problem and optionally filter its descriptions.

    Args:
        df: Combined benchmark results.
        name: Requested benchmark problem.
        description_include: Optional regular expression descriptions must match.
        description_exclude: Optional regular expression descriptions must not match.
        problem_size: Optional exact problem size.
        precision: Optional floating-point precision in bits.

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

    include_expression = _compile_description_filter(description_include, "--include")
    if include_expression is not None:
        mask = selected[DESCRIPTION].map(
            lambda value: include_expression.search(value) is not None
        )
        selected = selected.loc[mask].copy()
        if selected.empty:
            raise ValueError(
                f"Description include regex {description_include!r} matched no "
                f"rows for {resolved_name}."
            )

    exclude_expression = _compile_description_filter(description_exclude, "--exclude")
    if exclude_expression is not None:
        mask = selected[DESCRIPTION].map(
            lambda value: exclude_expression.search(value) is None
        )
        selected = selected.loc[mask].copy()
        if selected.empty:
            raise ValueError(
                f"Description exclude regex {description_exclude!r} excluded "
                f"all rows for {resolved_name}."
            )

    if problem_size is not None:
        selected = _filter_problem_size(selected, problem_size, resolved_name)

    if precision is not None:
        numeric_precision = pd.to_numeric(selected[PRECISION], errors="coerce")
        selected = selected.loc[numeric_precision.eq(precision)].copy()
        if selected.empty:
            available = sorted(
                pd.to_numeric(
                    df.loc[df[BENCHMARK_PROBLEM] == resolved_name, PRECISION],
                    errors="coerce",
                )
                .dropna()
                .unique()
            )
            raise ValueError(
                f"Precision {precision} has no rows for {resolved_name}. "
                f"Available precisions: {available}"
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
    non_zero_pp: bool = False,
    hardware_universe: Iterable[str] | None = None,
    application_universe: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate application efficiency and performance portability.

    Duplicate measurements are reduced to their median runtime. Application
    efficiencies are computed per hardware/workload and then arithmetically
    averaged over workloads. PP is the harmonic mean across all selected
    hardware. By default, an absent result contributes zero and makes PP zero.
    In non-zero PP mode, platforms with zero efficiency are omitted from the
    harmonic mean; the application-efficiency output still retains those zeros.

    Args:
        df: Selected benchmark rows.
        description_is_workload: Whether Description belongs to the workload
            key rather than the application label.
        non_zero_pp: Whether to exclude unsupported platforms from PP.
        hardware_universe: Optional complete platform set. This is used for
            per-size scaling so a platform with no row at one size contributes
            zero rather than disappearing from that size's PP calculation.
        application_universe: Optional complete application set. This is used
            for per-size calculations so an implementation with no row at one
            size receives zero rather than disappearing from that size.

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

    applications = sorted(
        set(application_universe)
        if application_universe is not None
        else set(runtimes[APPLICATION].unique())
    )
    hardware = sorted(
        set(hardware_universe)
        if hardware_universe is not None
        else set(runtimes[HARDWARE].unique())
    )
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
        """Calculate PP, optionally omitting unsupported platforms."""
        numeric = values.to_numpy(dtype=float)
        if non_zero_pp:
            numeric = numeric[numeric > 0.0]
            if len(numeric) == 0:
                return 0.0
            return float(len(numeric) / np.reciprocal(numeric).sum())
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


def calculate_metrics_by_size(
    df: pd.DataFrame,
    description_is_workload: bool,
    non_zero_pp: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate both metrics independently for each problem size.

    Args:
        df: Selected rows for one benchmark problem.
        description_is_workload: Whether Description belongs to the workload key.
        non_zero_pp: Whether unsupported platforms are excluded from PP.

    Returns:
        Efficiency and portability DataFrames that retain problem size.
    """
    numeric_sizes = pd.to_numeric(df[PROBLEM_SIZE], errors="coerce")
    sizes = sorted(numeric_sizes.dropna().unique())
    if not sizes:
        raise ValueError("No numeric problem sizes remain for the scaling plot.")

    platforms = sorted(df[HARDWARE].unique())
    applications = sorted(
        {
            _application_label(paradigm, description, description_is_workload)
            for paradigm, description in zip(df[PARADIGM], df[DESCRIPTION])
        }
    )
    efficiency_frames: list[pd.DataFrame] = []
    portability_frames: list[pd.DataFrame] = []
    for size in sizes:
        size_rows = df.loc[
            np.isclose(numeric_sizes.to_numpy(dtype=float), size, equal_nan=False)
        ]
        efficiency, portability = calculate_metrics(
            size_rows,
            description_is_workload,
            non_zero_pp=non_zero_pp,
            hardware_universe=platforms,
            application_universe=applications,
        )
        efficiency.insert(1, PROBLEM_SIZE, float(size))
        portability.insert(1, PROBLEM_SIZE, float(size))
        efficiency_frames.append(efficiency)
        portability_frames.append(portability)
    return (
        pd.concat(efficiency_frames, ignore_index=True),
        pd.concat(portability_frames, ignore_index=True),
    )


def calculate_scaling_metrics(
    df: pd.DataFrame,
    description_is_workload: bool,
    non_zero_pp: bool = False,
) -> pd.DataFrame:
    """Calculate performance portability independently for each problem size."""
    _, portability = calculate_metrics_by_size(
        df,
        description_is_workload,
        non_zero_pp=non_zero_pp,
    )
    return portability


def calculate_average_size_metrics(
    df: pd.DataFrame,
    description_is_workload: bool,
    non_zero_pp: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average application efficiency and PP independently over sizes.

    Both metrics are first calculated independently at every size, then
    arithmetically averaged. This gives every problem size equal weight and
    avoids taking the harmonic mean of already size-averaged efficiencies when
    ``--size average`` was explicitly requested.
    """
    per_size_efficiency, per_size_portability = calculate_metrics_by_size(
        df,
        description_is_workload,
        non_zero_pp=non_zero_pp,
    )
    efficiency = (
        per_size_efficiency.groupby([APPLICATION, HARDWARE], as_index=False)[
            APPLICATION_EFFICIENCY
        ]
        .mean()
        .sort_values([APPLICATION, HARDWARE])
    )
    portability = (
        per_size_portability.groupby(APPLICATION, as_index=False)[
            PERFORMANCE_PORTABILITY
        ]
        .mean()
        .sort_values(PERFORMANCE_PORTABILITY, ascending=False)
    )
    return efficiency, portability


def calculate_extreme_size_metrics(
    df: pd.DataFrame,
    description_is_workload: bool,
    mode: str,
    non_zero_pp: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select each application's best or worst metrics over problem sizes.

    Application efficiency is reduced independently for every application and
    hardware pair. Performance portability is reduced independently for every
    application, so both Cascade panels represent the requested size-axis
    extreme of the metric they display.

    Args:
        df: Selected rows for one benchmark problem.
        description_is_workload: Whether Description belongs to the workload key.
        mode: ``best`` for maxima or ``worst`` for minima.
        non_zero_pp: Whether unsupported platforms are excluded from PP.

    Returns:
        Application-efficiency and performance-portability extrema.

    Raises:
        ValueError: If mode is not ``best`` or ``worst``.
    """
    if mode not in {BEST_SIZE, WORST_SIZE}:
        raise ValueError(f"Unsupported size-extreme mode: {mode!r}")

    per_size_efficiency, per_size_portability = calculate_metrics_by_size(
        df,
        description_is_workload,
        non_zero_pp=non_zero_pp,
    )
    reduction = "max" if mode == BEST_SIZE else "min"
    efficiency = (
        per_size_efficiency.groupby([APPLICATION, HARDWARE], as_index=False)[
            APPLICATION_EFFICIENCY
        ]
        .agg(reduction)
        .sort_values([APPLICATION, HARDWARE])
    )
    portability = (
        per_size_portability.groupby(APPLICATION, as_index=False)[
            PERFORMANCE_PORTABILITY
        ]
        .agg(reduction)
        .sort_values(PERFORMANCE_PORTABILITY, ascending=False)
    )
    return efficiency, portability


def calculate_export_metrics(
    df: pd.DataFrame,
    description_is_workload: bool,
    non_zero_pp: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate export metrics without combining sizes or precisions.

    Returns one efficiency row per application, size, precision, and hardware,
    and one portability row per application, size, and precision. Additional
    rows with Problem Size ``average`` contain per-precision arithmetic means
    over the independently calculated size metrics.
    """
    dimensions = [PROBLEM_SIZE, PRECISION]
    platforms = sorted(df[HARDWARE].unique())
    applications = sorted(
        {
            _application_label(paradigm, description, description_is_workload)
            for paradigm, description in zip(df[PARADIGM], df[DESCRIPTION])
        }
    )
    efficiency_frames: list[pd.DataFrame] = []
    portability_frames: list[pd.DataFrame] = []
    for values, rows in df.groupby(dimensions, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        efficiency, portability = calculate_metrics(
            rows,
            description_is_workload,
            non_zero_pp=non_zero_pp,
            hardware_universe=platforms,
            application_universe=applications,
        )
        for position, (column, value) in enumerate(zip(dimensions, values), start=1):
            efficiency.insert(position, column, value)
            portability.insert(position, column, value)
        efficiency_frames.append(efficiency)
        portability_frames.append(portability)
    if not efficiency_frames:
        raise ValueError("No problem-size/precision groups remain for CSV export.")
    efficiency = pd.concat(efficiency_frames, ignore_index=True)
    portability = pd.concat(portability_frames, ignore_index=True)
    average_efficiency = (
        efficiency.groupby(
            [APPLICATION, PRECISION, HARDWARE],
            as_index=False,
            dropna=False,
        )[APPLICATION_EFFICIENCY]
        .mean()
        .sort_values([APPLICATION, PRECISION, HARDWARE])
    )
    average_efficiency.insert(1, PROBLEM_SIZE, AVERAGE_SIZE)
    average_portability = (
        portability.groupby(
            [APPLICATION, PRECISION],
            as_index=False,
            dropna=False,
        )[PERFORMANCE_PORTABILITY]
        .mean()
        .sort_values([APPLICATION, PRECISION])
    )
    average_portability.insert(1, PROBLEM_SIZE, AVERAGE_SIZE)
    return (
        pd.concat([efficiency, average_efficiency], ignore_index=True),
        pd.concat([portability, average_portability], ignore_index=True),
    )


def _slugify(value: str) -> str:
    """Convert a value into a compact filename component.

    Args:
        value: Arbitrary text.

    Returns:
        A filesystem-friendly lowercase slug.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "plot"


def resolve_output_path(
    output: Path | None,
    problems: list[str],
    mode: str,
) -> Path:
    """Resolve the plot output path and default PDF extension.

    Args:
        output: User-supplied path, or ``None``.
        problems: Resolved problem names.
        mode: Plot mode: ``cascade``, ``navchart``, or ``combined``.

    Returns:
        Output path with a suffix.
    """
    problem_slug = _slugify("_".join(problems))
    path = output or Path(f"{problem_slug}_{mode}.pdf")
    if not path.suffix:
        path = path.with_suffix(".pdf")
    return path


def _hardware_vendor(hardware: str) -> str:
    """Classify a hardware label using vendor regular expressions.

    Args:
        hardware: Hardware label from the benchmark CSV.

    Returns:
        ``NVIDIA``, ``AMD``, ``Intel``, or ``Other``.
    """
    for vendor, pattern in HARDWARE_VENDOR_PATTERNS.items():
        if pattern.search(hardware):
            return vendor
    return "Other"


def _hardware_colors(platforms: list[str]) -> dict[str, object]:
    """Assign vendor-family color shades to hardware platforms.

    NVIDIA hardware uses green shades, AMD red shades, Intel blue shades, and
    unmatched labels neutral gray shades.

    Args:
        platforms: Hardware labels.

    Returns:
        Color mapping keyed by hardware label.
    """
    palette_names = {
        "NVIDIA": "Greens",
        "AMD": "Reds",
        "Intel": "Blues",
        "Other": "Greys",
    }
    colors: dict[str, object] = {}
    for vendor, palette_name in palette_names.items():
        matches = sorted(
            platform for platform in platforms if _hardware_vendor(platform) == vendor
        )
        shades = sns.color_palette(palette_name, n_colors=len(matches) + 2)[2:]
        colors.update(dict(zip(matches, shades)))
    return colors


def _framework_colors(
    applications: list[str], remove_description: bool = False
) -> dict[str, object]:
    """Assign stable framework colors based on the requested tab20 scheme.

    Args:
        applications: Application/framework labels.
        remove_description: Whether bracketed variants use their base framework
            color so the shortened legend remains unambiguous.

    Returns:
        Color mapping keyed by application label.
    """
    known = sorted(
        (
            (_normalize_token(framework), color)
            for framework, color in FRAMEWORK_COLOR_MAP.items()
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    result: dict[str, object] = {}
    unmatched: list[str] = []
    for application in dict.fromkeys(applications):
        style_name = _display_application(application, remove_description)
        candidate_tokens = [
            _normalize_token(candidate)
            for candidate in _complexity_candidates(style_name)
        ]
        for framework_token, color in known:
            if any(framework_token in token for token in candidate_tokens):
                result[application] = color
                break
        else:
            unmatched.append(application)

    fallback = sns.color_palette("husl", n_colors=max(len(unmatched), 1))
    result.update(dict(zip(sorted(unmatched), fallback)))
    return result


def _problem_markers(problems: list[str]) -> dict[str, str]:
    """Assign the common dot marker to every benchmark problem.

    Args:
        problems: Resolved benchmark problem names.

    Returns:
        Marker mapping keyed by problem.
    """
    return {problem: "o" for problem in dict.fromkeys(problems)}


def _display_application(application: str, remove_description: bool) -> str:
    """Format an application label for a legend.

    Args:
        application: Full application label.
        remove_description: Whether to remove bracketed descriptions.

    Returns:
        Legend label.
    """
    if not remove_description:
        return application
    return re.sub(r"\[[^]]*]", "", application).strip()


def _application_legend_handles(
    applications: list[str],
    colors: dict[str, object],
    remove_description: bool,
) -> list[Line2D]:
    """Create framework-color legend handles.

    Args:
        applications: Ordered application labels.
        colors: Application color mapping.
        remove_description: Whether to hide bracketed descriptions.

    Returns:
        Matplotlib legend handles.
    """
    handles: list[Line2D] = []
    seen: set[tuple[str, tuple[float, ...]]] = set()
    ordered_applications = sorted(
        dict.fromkeys(applications),
        key=lambda application: _display_application(
            application, remove_description
        ).casefold(),
    )
    for application in ordered_applications:
        label = _display_application(application, remove_description)
        color = tuple(matplotlib.colors.to_rgba(colors[application]))
        key = (label, color)
        if key in seen:
            continue
        seen.add(key)
        handles.append(Line2D([0], [0], color=color, linewidth=3.0, label=label))
    return handles


def _problem_legend_handles(
    problems: list[str], markers: dict[str, str]
) -> list[Line2D]:
    """Create problem-marker legend handles.

    Args:
        problems: Ordered problem names.
        markers: Problem marker mapping.

    Returns:
        Matplotlib legend handles.
    """
    return [
        Line2D(
            [0],
            [0],
            color="black",
            marker=markers[problem],
            linestyle="None",
            markersize=LEGEND_MARKER_SIZE,
            markeredgewidth=1.5,
            label=problem,
        )
        for problem in dict.fromkeys(problems)
    ]


def plot_cascade(
    efficiency: pd.DataFrame,
    portability: pd.DataFrame,
    problem_title: str,
    remove_description: bool = False,
    navchart_data: pd.DataFrame | None = None,
    complexity_metric: str | None = None,
    scaling_data: pd.DataFrame | None = None,
    log_complexity: bool = False,
    log_size: bool = False,
    selected_size: float | str | None = None,
    show_legends: bool = True,
) -> plt.Figure:
    """Create a Cascade Plot following the P3 Analysis Library layout.

    Args:
        efficiency: Application efficiency by problem, application, and hardware.
        portability: Performance portability by problem and application.
        problem_title: Problem name or names used in the title.
        remove_description: Whether to hide bracketed legend descriptions.
        navchart_data: Optional complexity/PP data for combined mode.
        complexity_metric: Complexity column used in combined mode.
        scaling_data: Optional per-size PP data for combined mode.
        log_complexity: Whether the complexity axis is logarithmic.
        log_size: Whether the problem-size scaling axis is logarithmic.
        selected_size: Optional benchmark size or size-summary mode for the two
            upper panels.
        show_legends: Whether legends are embedded in the plot.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If combined-mode inputs are incomplete.
    """
    combined = navchart_data is not None
    if combined and (complexity_metric is None or scaling_data is None):
        raise ValueError(
            "Combined plotting requires complexity and per-size scaling data."
        )

    series = portability.sort_values(
        PERFORMANCE_PORTABILITY, ascending=False
    ).reset_index(drop=True)
    applications = list(dict.fromkeys(series[APPLICATION]))
    problems = list(dict.fromkeys(series[PROBLEM]))
    platforms = sorted(efficiency[HARDWARE].unique())
    colors = _framework_colors(applications, remove_description)
    markers = _problem_markers(problems)
    platform_colors = _hardware_colors(platforms)
    platform_labels = {
        platform: chr(ord("A") + index) if index < 26 else str(index + 1)
        for index, platform in enumerate(platforms)
    }

    series_count = len(series)
    figure_width = max(11.0, 7.5 + 0.25 * series_count)
    figure_height = max(7.5, 5.6 + 0.27 * series_count)
    figure = plt.figure(figsize=(figure_width, figure_height))
    right_ratio = 4.0 if combined else max(1.7, 0.22 * series_count)
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[4.5, max(1.5, 0.34 * series_count)],
        width_ratios=[5.5, right_ratio],
        hspace=0.3 if combined else 0.03,
        wspace=0.08,
    )
    efficiency_axis = figure.add_subplot(grid[0, 0])
    platform_axis = figure.add_subplot(grid[1, 0], sharex=efficiency_axis)
    portability_axis = figure.add_subplot(grid[0, 1], sharey=efficiency_axis)
    scaling_axis = (
        figure.add_subplot(grid[1, 1], sharey=efficiency_axis) if combined else None
    )

    for _, item in series.iterrows():
        problem = str(item[PROBLEM])
        application = str(item[APPLICATION])
        rows = efficiency.loc[
            (efficiency[PROBLEM] == problem)
            & (efficiency[APPLICATION] == application)
            & efficiency[APPLICATION_EFFICIENCY].gt(0.0)
        ].sort_values(APPLICATION_EFFICIENCY, ascending=False)
        ranks = np.arange(1, len(rows) + 1)
        efficiency_axis.plot(
            ranks,
            rows[APPLICATION_EFFICIENCY],
            color=colors[application],
            marker=markers[problem],
            linewidth=1.6,
            markersize=PLOT_MARKER_SIZE,
            markeredgewidth=1.5,
        )

    efficiency_axis.set_ylabel("Application Efficiency")
    efficiency_axis.set_ylim(0.0, 1.05)
    efficiency_axis.set_xlim(0.5, len(platforms) + 0.5)
    efficiency_axis.set_xticks(np.arange(1, len(platforms) + 1))
    efficiency_axis.tick_params(axis="x", labelbottom=False)
    efficiency_axis.grid(True, linestyle="--", alpha=0.45)
    plot_name = (
        f"{problem_title} {PP_SYMBOL} - Code Complexity Plot"
        if combined
        else f"{problem_title} {PP_SYMBOL} Cascade Plot"
    )
    figure.suptitle(plot_name, y=0.98)

    row_height = 1.0
    reversed_series = series.iloc[::-1].reset_index(drop=True)
    for row_index, item in reversed_series.iterrows():
        problem = str(item[PROBLEM])
        application = str(item[APPLICATION])
        rows = efficiency.loc[
            (efficiency[PROBLEM] == problem)
            & (efficiency[APPLICATION] == application)
            & efficiency[APPLICATION_EFFICIENCY].gt(0.0)
        ].sort_values(APPLICATION_EFFICIENCY, ascending=False)
        y = row_index * row_height
        platform_axis.plot(
            [0.5, len(platforms) + 0.5],
            [y + 0.5, y + 0.5],
            color=colors[application],
            marker=markers[problem],
            markersize=PLOT_MARKER_SIZE,
            markeredgewidth=1.5,
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
    platform_axis.set_ylim(0.0, max(series_count, 1))
    platform_axis.set_yticks([])
    platform_axis.grid(False)

    if combined:
        assert navchart_data is not None
        assert complexity_metric is not None
        for _, item in navchart_data.iterrows():
            problem = str(item[PROBLEM])
            application = str(item[APPLICATION])
            portability_axis.scatter(
                item[complexity_metric],
                item[PERFORMANCE_PORTABILITY],
                color=colors[application],
                marker=markers[problem],
                s=SCATTER_MARKER_AREA,
                linewidth=1.5,
                zorder=3,
            )
        portability_axis.set_xlabel(complexity_metric)
        if log_complexity:
            portability_axis.set_xscale("log")
        else:
            portability_axis.xaxis.set_major_formatter(EngFormatter())
        portability_axis.grid(True, linestyle="--", alpha=0.45)
    else:
        positions = np.arange(series_count)
        portability_axis.bar(
            positions,
            series[PERFORMANCE_PORTABILITY],
            color="white",
            edgecolor=[colors[name] for name in series[APPLICATION]],
            linewidth=1.5,
        )
        for position, (_, item) in zip(positions, series.iterrows()):
            portability_axis.scatter(
                position,
                item[PERFORMANCE_PORTABILITY],
                color=colors[str(item[APPLICATION])],
                marker=markers[str(item[PROBLEM])],
                s=SCATTER_MARKER_AREA,
                linewidth=1.5,
                zorder=3,
            )
        portability_axis.set_xticks([])
        portability_axis.grid(True, axis="y", linestyle="--", alpha=0.45)
    portability_axis.yaxis.tick_right()
    portability_axis.yaxis.set_label_position("right")
    portability_axis.set_ylabel(PP_SYMBOL)

    if combined:
        assert scaling_axis is not None
        assert scaling_data is not None
        for (problem, application), rows in scaling_data.groupby(
            [PROBLEM, APPLICATION], sort=False
        ):
            rows = rows.sort_values(PROBLEM_SIZE)
            scaling_axis.plot(
                rows[PROBLEM_SIZE],
                rows[PERFORMANCE_PORTABILITY],
                color=colors[str(application)],
                marker=markers[str(problem)],
                linewidth=1.5,
                markersize=PLOT_MARKER_SIZE,
                markeredgewidth=1.5,
            )
        scaling_axis.set_xlabel("Problem Size")
        if log_size:
            scaling_axis.set_xscale("log")
        scaling_axis.yaxis.tick_right()
        scaling_axis.yaxis.set_label_position("right")
        scaling_axis.set_ylabel(PP_SYMBOL)
        scaling_axis.grid(True, linestyle="--", alpha=0.45)

    if show_legends:
        application_handles = _application_legend_handles(
            applications, colors, remove_description
        )
        app_legend = figure.legend(
            handles=application_handles,
            title="Paradigm",
            loc="center left",
            bbox_to_anchor=(0.72, 0.55),
            frameon=True,
        )
        figure.add_artist(app_legend)

        if len(problems) > 1:
            problem_legend = figure.legend(
                handles=_problem_legend_handles(problems, markers),
                title="Problem",
                loc="lower left",
                bbox_to_anchor=(0.72, 0.08),
                frameon=True,
            )
            figure.add_artist(problem_legend)

    platform_handles = [
        Patch(
            facecolor=platform_colors[name],
            edgecolor="black",
            label=f"{platform_labels[name]}: {name}",
        )
        for name in platforms
    ]
    hardware_columns = min(2, max(len(platforms), 1))
    hardware_rows = (len(platforms) + hardware_columns - 1) // hardware_columns
    bottom_margin = 0.23 + 0.045 * max(hardware_rows - 1, 0)
    if show_legends:
        figure.legend(
            handles=platform_handles,
            title="Device",
            loc="lower center",
            bbox_to_anchor=(0.35, 0.01),
            ncol=hardware_columns,
            frameon=True,
        )
        figure.subplots_adjust(right=0.68, bottom=bottom_margin, top=0.9)
    else:
        figure.subplots_adjust(right=0.97, bottom=0.1, top=0.9)

    if selected_size is not None:
        upper_left = efficiency_axis.get_position()
        upper_right = portability_axis.get_position()
        if selected_size == AVERAGE_SIZE:
            size_note = f"{PP_SYMBOL} and $e_A$ averaged over benchmark sizes"
        elif selected_size in {BEST_SIZE, WORST_SIZE}:
            size_note = (
                f"{PP_SYMBOL} and $e_A$ use the {selected_size} value over "
                "benchmark sizes"
            )
        else:
            size_note = (
                f"{PP_SYMBOL} and $e_A$ plotted for benchmark size = "
                f"{float(selected_size):g}"
            )
        figure.text(
            (upper_left.x0 + upper_right.x1) / 2.0,
            max(upper_left.y1, upper_right.y1) + 0.012,
            size_note,
            ha="center",
            va="bottom",
            fontsize=matplotlib.rcParams["axes.titlesize"],
        )
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
    result["Halstead Vocabulary"] = vocabulary
    result["Halstead Program Length"] = length
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
            column for column in df.columns if column.casefold() == requested.casefold()
        ]
        if len(direct) != 1:
            raise ValueError(
                f"Unknown complexity metric {requested!r}. Use one of: "
                "SLOC, Halstead vocabulary, Halstead program length, Halstead "
                "volume, Halstead difficulty, Halstead effort."
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
    additive: bool,
) -> tuple[pd.DataFrame, str]:
    """Load, select, calculate, and optionally normalize complexity data.

    Args:
        path: Complexity CSV path.
        problem_query: User-supplied benchmark problem query.
        metric_request: Requested metric or alias.
        normalize: Whether to express values relative to CPP.
        additive: Whether to subtract the CPP value from every value.

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

    display_metric = "Source Lines of Code" if metric == "SLOC" else metric
    label = f"{display_metric} [absolute]"
    if normalize or additive:
        tokens = selected[COMPLEXITY_FRAMEWORK].map(_normalize_token)
        baseline_rows = selected.loc[tokens.isin({"cpp", "cplusplus", "cpu"}), metric]
        if len(baseline_rows) != 1:
            raise ValueError(
                f"--{'normalize' if normalize else 'additive'} requires exactly "
                "one CPP complexity row "
                f"for {problem}; found {len(baseline_rows)}."
            )
        baseline = float(baseline_rows.iloc[0])
        if normalize:
            if baseline <= 0.0:
                raise ValueError(
                    "--normalize requires a positive CPP complexity score."
                )
            selected[metric] = selected[metric] / baseline * 100.0
            label = f"{display_metric} [normalized]"
            logger.debug(f"Normalized {metric} to CPP baseline {baseline:g}")
        else:
            selected[metric] = selected[metric] - baseline
            label = f"{display_metric} [additive]"
            logger.debug(f"Subtracted CPP {metric} baseline {baseline:g}")
    selected = selected.rename(columns={metric: label})
    logger.info(f"Loaded {len(selected)} complexity rows for {problem}")
    return selected, label


def load_complexity_export_data(
    path: Path,
    problem_query: str,
) -> pd.DataFrame:
    """Load every numeric complexity metric and derive Halstead metrics.

    Source metric columns such as SLOC and the primitive Halstead counts are
    retained. When all four primitive counts are available, all supported
    derived Halstead metrics are calculated even if they were not requested
    for the plot.

    Args:
        path: Complexity CSV path.
        problem_query: User-supplied benchmark problem query.

    Returns:
        Median numeric complexity metrics grouped by Framework.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the input schema or Halstead counts are invalid, or no
            numeric metric columns are available.
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

    halstead_counts = {"n1", "n2", "N1", "N2"}
    if halstead_counts.issubset(selected.columns):
        selected = _calculate_halstead_columns(selected, path)

    metric_columns: list[str] = []
    identity_columns = {COMPLEXITY_NAME, COMPLEXITY_FRAMEWORK}
    for column in selected.columns:
        if column in identity_columns:
            continue
        numeric = pd.to_numeric(selected[column], errors="coerce")
        if numeric.notna().any():
            selected[column] = numeric
            metric_columns.append(column)
    if not metric_columns:
        raise ValueError(f"{path} contains no numeric complexity metrics.")

    result = selected.groupby(COMPLEXITY_FRAMEWORK, as_index=False)[
        metric_columns
    ].median()
    logger.info(f"Loaded {len(metric_columns)} export complexity metrics for {problem}")
    return result


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


def plot_navchart(
    data: pd.DataFrame,
    metric: str,
    problem_title: str,
    remove_description: bool = False,
    log_complexity: bool = False,
    show_legends: bool = True,
) -> plt.Figure:
    """Create a Navchart of complexity and performance portability.

    Args:
        data: Matched problem, application, PP, and complexity data.
        metric: Complexity metric column and x-axis label.
        problem_title: Problem name or names used in the title.
        remove_description: Whether to hide bracketed legend descriptions.
        log_complexity: Whether the complexity axis is logarithmic.
        show_legends: Whether legends are embedded in the plot.

    Returns:
        The Matplotlib figure.
    """
    applications = list(dict.fromkeys(data[APPLICATION]))
    problems = list(dict.fromkeys(data[PROBLEM]))
    colors = _framework_colors(applications, remove_description)
    markers = _problem_markers(problems)
    figure, axis = plt.subplots(figsize=(9.5, 7.2))
    for _, row in data.iterrows():
        application = str(row[APPLICATION])
        problem = str(row[PROBLEM])
        marker = markers[problem]
        marker_options = (
            {"linewidth": 1.2}
            if marker in {"x", "+", "1", "2", "3", "4", "|", "_"}
            else {"edgecolor": "black", "linewidth": 0.5}
        )
        axis.scatter(
            row[metric],
            row[PERFORMANCE_PORTABILITY],
            color=colors[application],
            marker=marker,
            s=SCATTER_MARKER_AREA,
            zorder=3,
            **marker_options,
        )

    axis.set_xlabel(metric)
    axis.set_ylabel(PP_SYMBOL)
    axis.set_ylim(0.0, 1.05)
    figure.suptitle(f"{problem_title} {PP_SYMBOL} - Code Complexity")
    if log_complexity:
        axis.set_xscale("log")
    else:
        axis.xaxis.set_major_formatter(EngFormatter())
    axis.grid(True, linestyle="--", alpha=0.45)
    if show_legends:
        application_legend = figure.legend(
            handles=_application_legend_handles(
                applications, colors, remove_description
            ),
            title="Paradigm",
            loc="center left",
            bbox_to_anchor=(0.72, 0.55),
            frameon=True,
        )
        figure.add_artist(application_legend)
        if len(problems) > 1:
            figure.legend(
                handles=_problem_legend_handles(problems, markers),
                title="Problem",
                loc="lower center",
                bbox_to_anchor=(0.34, 0.01),
                ncol=len(problems),
                frameon=True,
            )
        figure.subplots_adjust(right=0.68, bottom=0.28 if len(problems) > 1 else 0.11)
    else:
        figure.subplots_adjust(right=0.96, bottom=0.11)
    return figure


def create_separate_legend(
    applications: list[str],
    problems: list[str],
    platforms: list[str],
    remove_description: bool = False,
) -> plt.Figure:
    """Create a standalone four-column paradigm/device legend.

    Paradigms are placed at the top and devices at the bottom. A problem-marker
    section is included between them when more than one problem is plotted.

    Args:
        applications: Application labels in display order.
        problems: Problem names in display order.
        platforms: Hardware labels in display order.
        remove_description: Whether bracketed descriptions are hidden.

    Returns:
        A standalone legend figure.
    """
    colors = _framework_colors(applications, remove_description)
    markers = _problem_markers(problems)
    platform_colors = _hardware_colors(platforms)
    platform_labels = {
        platform: chr(ord("A") + index) if index < 26 else str(index + 1)
        for index, platform in enumerate(platforms)
    }
    paradigm_handles = _application_legend_handles(
        applications, colors, remove_description
    )
    device_handles = [
        Patch(
            facecolor=platform_colors[name],
            edgecolor="black",
            label=f"{platform_labels[name]}: {name}",
        )
        for name in platforms
    ]

    paradigm_rows = max(1, (len(paradigm_handles) + 3) // 4)
    device_rows = max(1, (len(device_handles) + 3) // 4)
    problem_rows = 1 if len(problems) > 1 else 0
    height = 1.0 + 0.55 * (paradigm_rows + device_rows + problem_rows)
    figure = plt.figure(figsize=(13.0, height))
    figure.legend(
        handles=paradigm_handles,
        title="Paradigm",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=True,
    )
    if len(problems) > 1:
        figure.legend(
            handles=_problem_legend_handles(problems, markers),
            title="Problem",
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            ncol=4,
            frameon=True,
        )
    figure.legend(
        handles=device_handles,
        title="Device",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=True,
    )
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


def add_complexity_to_export(
    data: pd.DataFrame,
    complexity: pd.DataFrame,
) -> pd.DataFrame:
    """Add all matching complexity values to an export DataFrame.

    Export rows are preserved when no complexity framework matches an
    application; the metrics are left empty for those rows.

    Args:
        data: Application-efficiency or performance-portability data.
        complexity: Complexity values keyed by Framework.

    Returns:
        A copy of ``data`` with every complexity metric after Application.
    """
    metric_columns = [
        column for column in complexity.columns if column != COMPLEXITY_FRAMEWORK
    ]
    overlapping = set(metric_columns) & set(data.columns)
    if overlapping:
        raise ValueError(
            "Complexity metric columns duplicate export columns: "
            f"{sorted(overlapping)}"
        )
    lookup = {
        _normalize_token(row[COMPLEXITY_FRAMEWORK]): {
            column: float(row[column]) for column in metric_columns
        }
        for _, row in complexity.iterrows()
    }
    matched_values: list[dict[str, float] | None] = []
    unmatched: set[str] = set()
    for application_value in data[APPLICATION]:
        application = str(application_value)
        value = next(
            (
                lookup[token]
                for candidate in _complexity_candidates(application)
                if (token := _normalize_token(candidate)) in lookup
            ),
            None,
        )
        matched_values.append(value)
        if value is None:
            unmatched.add(application)

    if unmatched:
        logger.warning(
            "No complexity value for exported application(s): "
            + ", ".join(sorted(unmatched))
        )
    result = data.copy()
    insert_position = result.columns.get_loc(APPLICATION) + 1
    for offset, column in enumerate(metric_columns):
        result.insert(
            insert_position + offset,
            column,
            [np.nan if values is None else values[column] for values in matched_values],
        )
    return result


def append_cpp_complexity_row(
    data: pd.DataFrame,
    complexity: pd.DataFrame,
    problem: str,
) -> pd.DataFrame:
    """Append the complexity-only CPP baseline to an export DataFrame.

    CPP is the complexity reference implementation but has no benchmark
    measurements. Its size, precision, hardware, and performance metric cells
    therefore remain empty in both exported datasets.
    """
    cpp_tokens = {"cpp", "cplusplus", "cpu"}
    cpp_rows = complexity.loc[
        complexity[COMPLEXITY_FRAMEWORK]
        .astype(str)
        .map(_normalize_token)
        .isin(cpp_tokens)
    ]
    if cpp_rows.empty:
        logger.warning(f"No CPP complexity row available for {problem}")
        return data
    if len(cpp_rows) > 1:
        raise ValueError(
            f"Expected at most one CPP complexity row for {problem}; "
            f"found {len(cpp_rows)}."
        )

    cpp = cpp_rows.iloc[0]
    cpp_application = str(cpp[COMPLEXITY_FRAMEWORK])
    if data[APPLICATION].astype(str).map(_normalize_token).eq(
        _normalize_token(cpp_application)
    ).any():
        return data

    row: dict[str, object] = {column: np.nan for column in data.columns}
    row[PROBLEM] = problem
    row[APPLICATION] = cpp_application
    for column in complexity.columns:
        if column != COMPLEXITY_FRAMEWORK and column in row:
            row[column] = cpp[column]
    return pd.concat([data, pd.DataFrame([row])], ignore_index=True)


def export_metrics_to_csv(
    efficiency: pd.DataFrame,
    portability: pd.DataFrame,
    output: Path,
) -> tuple[Path, Path]:
    """Export metric DataFrames next to the plot using its filename prefix.

    Args:
        efficiency: Application-efficiency data to export.
        portability: Performance-portability data to export.
        output: Resolved plot path whose stem supplies the filename prefix.

    Returns:
        The application-efficiency and performance-portability CSV paths.
    """
    efficiency_output = output.with_name(f"{output.stem}_application_efficiency.csv")
    portability_output = output.with_name(f"{output.stem}_performance_portability.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    efficiency.to_csv(efficiency_output, index=False)
    portability.to_csv(portability_output, index=False)
    logger.success(f"Wrote application efficiency: {efficiency_output.resolve()}")
    logger.success(f"Wrote performance portability: {portability_output.resolve()}")
    return efficiency_output, portability_output


def main() -> int:
    """Run the P3 analysis command-line program.

    Returns:
        Process exit status: zero on success, one on failure.
    """
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")

    try:
        if args.chart in {"navchart", "combined"} and args.complexity is None:
            raise ValueError(f"--chart {args.chart} requires --complexity.")
        if (args.normalize or args.additive) and (
            args.complexity is None or args.chart == "cascade"
        ):
            raise ValueError(
                "--normalize/--additive are only valid for navchart/combined "
                "with --complexity."
            )
        if args.log_complexity and args.chart == "cascade":
            raise ValueError(
                "--log-complexity is only valid for navchart/combined charts."
            )
        if args.log_size and args.chart != "combined":
            raise ValueError("--log-size is only valid for combined charts.")
        if (
            args.chart == "cascade"
            and args.complexity is not None
            and not args.export_to_csv
        ):
            logger.warning("Ignoring --complexity for --chart cascade.")

        benchmark_data = load_benchmark_csvs(args.csv_files)
        efficiency_frames: list[pd.DataFrame] = []
        portability_frames: list[pd.DataFrame] = []
        export_efficiency_frames: list[pd.DataFrame] = []
        export_portability_frames: list[pd.DataFrame] = []
        scaling_frames: list[pd.DataFrame] = []
        problem_pairs: list[tuple[str, str]] = []

        for problem_query in (args.name,):
            numeric_size = args.size if isinstance(args.size, float) else None
            selection_size = None if args.chart == "combined" else numeric_size
            all_size_rows, resolved_problem, description_is_workload = (
                select_problem_rows(
                    benchmark_data,
                    problem_query,
                    args.description_include,
                    args.description_exclude,
                    selection_size,
                    args.precision,
                )
            )
            problem_pairs.append((problem_query, resolved_problem))

            selected = all_size_rows
            if args.chart == "combined" and numeric_size is not None:
                selected = _filter_problem_size(
                    all_size_rows, numeric_size, resolved_problem
                )

            if args.size == AVERAGE_SIZE:
                problem_efficiency, problem_portability = (
                    calculate_average_size_metrics(
                        selected,
                        description_is_workload,
                        non_zero_pp=args.non_zero_pp,
                    )
                )
            elif args.size in {BEST_SIZE, WORST_SIZE}:
                problem_efficiency, problem_portability = (
                    calculate_extreme_size_metrics(
                        selected,
                        description_is_workload,
                        args.size,
                        non_zero_pp=args.non_zero_pp,
                    )
                )
            else:
                problem_efficiency, problem_portability = calculate_metrics(
                    selected,
                    description_is_workload,
                    non_zero_pp=args.non_zero_pp,
                )
            problem_efficiency.insert(0, PROBLEM, resolved_problem)
            problem_portability.insert(0, PROBLEM, resolved_problem)
            efficiency_frames.append(problem_efficiency)
            portability_frames.append(problem_portability)

            if args.export_to_csv:
                export_rows, export_problem, export_description_is_workload = (
                    select_problem_rows(
                        benchmark_data,
                        problem_query,
                        args.description_include,
                        args.description_exclude,
                        None,
                        None,
                    )
                )
                export_efficiency, export_portability = calculate_export_metrics(
                    export_rows,
                    export_description_is_workload,
                    non_zero_pp=args.non_zero_pp,
                )
                export_efficiency.insert(0, PROBLEM, export_problem)
                export_portability.insert(0, PROBLEM, export_problem)
                export_efficiency_frames.append(export_efficiency)
                export_portability_frames.append(export_portability)
            if args.chart == "combined":
                problem_scaling = calculate_scaling_metrics(
                    all_size_rows,
                    description_is_workload,
                    non_zero_pp=args.non_zero_pp,
                )
                problem_scaling.insert(0, PROBLEM, resolved_problem)
                scaling_frames.append(problem_scaling)

        efficiency = pd.concat(efficiency_frames, ignore_index=True)
        portability = pd.concat(portability_frames, ignore_index=True)
        problems = [resolved for _, resolved in problem_pairs]
        problem_title = " + ".join(problems)
        mode = args.chart
        output = resolve_output_path(args.output, problems, mode)

        navchart_data: pd.DataFrame | None = None
        metric: str | None = None
        export_efficiency: pd.DataFrame | None = None
        export_portability: pd.DataFrame | None = None
        if args.export_to_csv:
            export_efficiency = pd.concat(export_efficiency_frames, ignore_index=True)
            export_portability = pd.concat(export_portability_frames, ignore_index=True)

        if args.chart in {"navchart", "combined"}:
            assert args.complexity is not None
            navchart_frames: list[pd.DataFrame] = []
            for problem_query, resolved_problem in problem_pairs:
                complexity, current_metric = load_complexity_data(
                    args.complexity,
                    problem_query,
                    args.complexity_metric,
                    args.normalize,
                    args.additive,
                )
                if metric is not None and current_metric != metric:
                    raise ValueError(
                        "Complexity metric labels differ between problems: "
                        f"{metric!r} and {current_metric!r}."
                    )
                metric = current_metric
                problem_pp = portability.loc[
                    portability[PROBLEM] == resolved_problem
                ].drop(columns=PROBLEM)
                problem_navchart = merge_portability_complexity(
                    problem_pp, complexity, current_metric
                )
                problem_navchart.insert(0, PROBLEM, resolved_problem)
                navchart_frames.append(problem_navchart)

            navchart_data = pd.concat(navchart_frames, ignore_index=True)
            logger.debug(f"Navchart data:\n{navchart_data.to_string(index=False)}")
            assert metric is not None
            if args.log_complexity and navchart_data[metric].le(0.0).any():
                raise ValueError(
                    "--log-complexity requires all plotted complexity values "
                    "to be positive."
                )

        if args.export_to_csv and args.complexity is not None:
            assert export_efficiency is not None
            assert export_portability is not None
            enriched_efficiency_frames: list[pd.DataFrame] = []
            enriched_portability_frames: list[pd.DataFrame] = []
            for problem_query, resolved_problem in problem_pairs:
                export_complexity = load_complexity_export_data(
                    args.complexity, problem_query
                )
                portability_mask = export_portability[PROBLEM] == resolved_problem
                problem_export_portability = add_complexity_to_export(
                    export_portability.loc[portability_mask].copy(),
                    export_complexity,
                )
                efficiency_mask = export_efficiency[PROBLEM] == resolved_problem
                problem_export_efficiency = add_complexity_to_export(
                    export_efficiency.loc[efficiency_mask].copy(),
                    export_complexity,
                )
                enriched_portability_frames.append(
                    append_cpp_complexity_row(
                        problem_export_portability,
                        export_complexity,
                        resolved_problem,
                    )
                )
                enriched_efficiency_frames.append(
                    append_cpp_complexity_row(
                        problem_export_efficiency,
                        export_complexity,
                        resolved_problem,
                    )
                )
            export_efficiency = pd.concat(
                enriched_efficiency_frames, ignore_index=True, sort=False
            )
            export_portability = pd.concat(
                enriched_portability_frames, ignore_index=True, sort=False
            )

        if args.chart == "combined":
            assert navchart_data is not None
            assert metric is not None
            scaling_data = pd.concat(scaling_frames, ignore_index=True)
            if args.log_size and scaling_data[PROBLEM_SIZE].le(0.0).any():
                raise ValueError(
                    "--log-size requires all plotted problem sizes to be positive."
                )
            figure = plot_cascade(
                efficiency,
                portability,
                problem_title,
                remove_description=args.remove_description,
                navchart_data=navchart_data,
                complexity_metric=metric,
                scaling_data=scaling_data,
                log_complexity=args.log_complexity,
                log_size=args.log_size,
                selected_size=args.size,
                show_legends=not args.legend,
            )
        elif args.chart == "navchart":
            assert navchart_data is not None
            assert metric is not None
            figure = plot_navchart(
                navchart_data,
                metric,
                problem_title,
                remove_description=args.remove_description,
                log_complexity=args.log_complexity,
                show_legends=not args.legend,
            )
        else:
            figure = plot_cascade(
                efficiency,
                portability,
                problem_title,
                remove_description=args.remove_description,
                selected_size=args.size,
                show_legends=not args.legend,
            )

        save_figure(figure, output)
        if args.export_to_csv:
            assert export_efficiency is not None
            assert export_portability is not None
            export_identity_columns = {
                PROBLEM,
                APPLICATION,
                PROBLEM_SIZE,
                PRECISION,
                PERFORMANCE_PORTABILITY,
            }
            complexity_columns = [
                column
                for column in export_portability.columns
                if column not in export_identity_columns
            ]
            export_efficiency = export_efficiency[
                [
                    PROBLEM,
                    APPLICATION,
                    PROBLEM_SIZE,
                    PRECISION,
                    *complexity_columns,
                    HARDWARE,
                    APPLICATION_EFFICIENCY,
                ]
            ]
            export_portability = export_portability[
                [
                    PROBLEM,
                    APPLICATION,
                    PROBLEM_SIZE,
                    PRECISION,
                    *complexity_columns,
                    PERFORMANCE_PORTABILITY,
                ]
            ]
            export_metrics_to_csv(export_efficiency, export_portability, output)
        if args.legend:
            legend_source = navchart_data if args.chart == "navchart" else portability
            assert legend_source is not None
            legend_applications = list(
                dict.fromkeys(legend_source[APPLICATION].astype(str))
            )
            legend_figure = create_separate_legend(
                legend_applications,
                problems,
                sorted(efficiency[HARDWARE].astype(str).unique()),
                remove_description=args.remove_description,
            )
            legend_output = output.with_name(f"{output.stem}_legend.pdf")
            save_figure(legend_figure, legend_output)
        return 0
    except (FileNotFoundError, OSError, ValueError) as error:
        logger.error(str(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
