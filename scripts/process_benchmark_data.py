#!python3
"""
Benchmarking Script
===================

This script is designed to facilitate the benchmarking of executable files and process their results to generate plots.
It supports searching for executable files based on a directory and pattern, running these executables to obtain
benchmarking reports, and finally compiling these reports into visual plots and CSV files.

Usage:
------
1. Directly run benchmarks and generate plots and CSV:
    $ python script.py -p /path/to/search/ -r "*.out" -o output.pdf

2. Only search for existing report files and generate plots and CSV:
    $ python script.py --report -p /path/to/search/ -r "*.json" -o output.pdf

Arguments:
----------
--report:
    If set, the script will search for existing report files instead of running the benchmarking targets.

-p, --path:
    List of paths to search for files. Default is the current working directory.

-r, --regex:
    Regex pattern to match files.

-o, --output:
    Output file name for the plots. Default is `yyyy-mm-dd_hh-mm_Benchmark_Result.pdf`.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

# Define Column Names
BENCHMARK_PROBLEM = "Benchmark Problem"
PARADIGM = "Paradigm"
DESCRIPTION = "Description"
PRECISION = "Precision"
PROBLEM_SIZE = "Problem Size"
ITERATIONS = "Iterations"
WALL_CLOCK_TIME = "Wall Clock Time"
CPU_TIME = "CPU Time"
KERNEL_TIME = "Kernel Time"
POSITION_UPDATE_TIME = "Position Update Time"
VELOCITY_UPDATE_TIME = "Velocity Update Time"
FORCE_UPDATE_TIME = "Force Update Time"
TIME_UNIT = "Time Unit"

COLUMN_LIST = [
    BENCHMARK_PROBLEM,
    PARADIGM,
    DESCRIPTION,
    PRECISION,
    PROBLEM_SIZE,
    ITERATIONS,
    WALL_CLOCK_TIME,
    CPU_TIME,
    KERNEL_TIME,
    POSITION_UPDATE_TIME,
    VELOCITY_UPDATE_TIME,
    FORCE_UPDATE_TIME,
    TIME_UNIT,
]
"""All columns in the Output CSV File"""


MERGED_PARADIGM_DESCRIPTION = "Paradigm[Description]"
"""Required for plots"""


def find_files(search_dir: list[Path] | Path, pattern: str | list[str]) -> list[Path]:
    """Searches for files given a search directory and a glob pattern.
    Args:
        search_dir: Directory/ Directories to search for files.
        pattern: Pattern to match files.

    Returns:
        A list of file paths that match the given pattern.
    """
    cmake_targets = []
    if isinstance(search_dir, Path):
        search_dir = [search_dir]
    if isinstance(pattern, str):
        pattern = [pattern]
    logger.info(
        f"Recursive searching for files in {search_dir} according to pattern: {pattern}"
    )
    for directory in search_dir:
        for regex_pattern in pattern:
            for file in directory.rglob(regex_pattern):
                if file.is_file():
                    cmake_targets.append(file.resolve())
                    logger.debug(f"Append file {file}")
    logger.info(f"Found {len(cmake_targets)} files")
    return cmake_targets


def run_benchmarks(executable_targets: list[Path]) -> list[Path]:
    """Runs the executable targets containing outputting a JSON report.
    This method assumes that all executables use the Google Benchmark framework.

    Args:
        executable_targets: A list of Path objects representing the executable files to be benchmarked.

    Returns:
        A list of Path objects representing the report files generated for each benchmarked executable.
    """
    report_files = []
    for target in executable_targets:
        output_file = Path(f"{target.name}_report.json")
        report_files.append(output_file)
        if output_file.exists():
            logger.warning(f"Report for {target} already exists, skipping!")
            continue
        try:
            logger.info(f"Benchmarking {target} started!")
            subprocess.run(
                f"{target} --benchmark_out={output_file.name}",
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info(f"Benchmarking {target} finished!")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to execute {target}: {e}")
    return report_files


def load_reports(report_files: list[Path]) -> pd.DataFrame:
    """Gets a list of paths pointing to JSON benchmarking reports. Every report
    is read and normalized using the JSON library before being transferred to a
    `pandas.DataFrame`.

    Args:
        report_files: List of Path objects pointing to JSON report files to be loaded.

    Returns:
        concatenated DataFrame containing the processed benchmark data from all report files.
    """
    data = []
    logger.info("Loading all report json files")
    for file_path in report_files:
        logger.debug(f"Loading report file {file_path}")
        with open(file_path, "r") as file:
            json_data = json.load(file)

        # Parse Benchmarking Context including the float type and the utilized paradigm (every report can only have one underlying paradigm)
        float_type = json_data["context"]["float_type"]
        paradigm = json_data["context"]["paradigm"]

        # Create the DataFrame with the benchmarking data
        df = pd.DataFrame(json_data["benchmarks"])

        # Include the paradigm name and the float type in the DataFrame
        df[PARADIGM] = paradigm
        df[PRECISION] = float_type

        # Google Benchmark puts the problem size in the name, so we need to split it out
        name_size_split = df["name"].str.split("/", expand=True)

        # By OUR convention, we append a description to the benchmark name to further describe it (e.g. "cubals" or "shared memory")
        details_split = name_size_split[0].str.split("-", expand=True)

        # Remove NaN values from the DataFrame
        df = df.dropna(subset=["iterations"])

        if len(name_size_split.columns) > 1:
            # Include the problem size in the DataFrame (as parsed from the name)
            df[PROBLEM_SIZE] = name_size_split[1].astype(int)
        elif "NumFaces" in df.columns:
            # In case of the polyhedral benchmark, there is no problem size, it is a user-defined counter of the amount of faces
            df[PROBLEM_SIZE] = df["NumFaces"].astype(int)

        df[BENCHMARK_PROBLEM] = details_split[0].astype(str)
        if len(details_split.columns) > 1:
            # Include the description in the DataFrame (as parsed from the name by our `-` convention)
            df[DESCRIPTION] = details_split[1].astype(str)
        else:
            df[DESCRIPTION] = ""

        # Include columns specific to NBody (to facilitate DataFrame merging later), but leave them empty
        for col in [
            "kernel_time",
            "position_update_reset",
            "velocity_update",
            "force_update",
        ]:
            if col not in df.columns:
                df[col] = None

        # Rename columns to match our convention
        df.rename(
            columns={
                "real_time": WALL_CLOCK_TIME,
                "cpu_time": CPU_TIME,
                "time_unit": TIME_UNIT,
                "iterations": ITERATIONS,
                "kernel_time": KERNEL_TIME,
                "position_update_reset": POSITION_UPDATE_TIME,
                "velocity_update": VELOCITY_UPDATE_TIME,
                "force_update": FORCE_UPDATE_TIME,
            },
            inplace=True,
        )

        # Rearrange the column in the DataFrame
        df = df[COLUMN_LIST]
        data.append(df)
    logger.info(f"Loaded {len(data)} report json files")
    return pd.concat(data)


def plot_linechart(
    benchmark_df: pd.DataFrame,
    save_path: Path | None = None,
    time_unit: str = "nanoseconds",
    selected_runtime: Literal["kernel_time", "wall_time", "cpu_time"] = "wall_time",
    use_merged_paradigm_description: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Generates a log-log plot from the given benchmark DataFrame. and stores the result
    into a file if one is specified

    Args:
        benchmark_df: Aapandas DataFrame containing benchmark data with columns for 'Problem Size', 'Wall Runtime', 'Framework', 'Precision', and 'Name'.
        save_path (Optional): a Path object specifying the location to save the resultant plot PDF.
        time_unit (Optional): a string specifying the unit of time to use for the y-axis. Defaults to "nanoseconds".
        selected_runtime (Optional): a string specifying the time to use for the plot. Defaults to "wall_time".

    Returns:
        A tuple containing the matplotlib Figure and Axes objects for the generated plot.
    """
    df = benchmark_df.copy()
    runtime = {
        "wall_time": WALL_CLOCK_TIME,
        "cpu_time": CPU_TIME,
        "kernel_time": KERNEL_TIME,
    }[selected_runtime]
    name = PARADIGM

    if use_merged_paradigm_description:
        df[MERGED_PARADIGM_DESCRIPTION] = df.apply(
            lambda r: (
                r[PARADIGM]
                if r[DESCRIPTION] == ""
                else f"{r[PARADIGM]}[{r[DESCRIPTION]}]"
            ),
            axis=1,
        )
        name = MERGED_PARADIGM_DESCRIPTION

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        x=PROBLEM_SIZE,
        y=runtime,
        hue=name,
        hue_order=df[name].unique().sort(),
        data=df,
        marker="o",
        ax=ax,
        palette="tab20",
        style=PRECISION,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(f"{runtime} [{time_unit}]")
    ax.set_xlabel("Problem Size [1]")
    ax.set_title(
        f'Runtime in {time_unit} vs. Problem Size $N$ for {df["Name"].unique()[0]}'
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        logger.info(f"Saved results to file {save_path}")
    return fig, ax


def plot_barchart(
    benchmark_df: pd.DataFrame,
    save_path: Path | None = None,
    time_unit: str = "nanoseconds",
    selected_runtime: list[Literal["kernel_time", "wall_time", "cpu_time"]] = [
        "wall_time",
        "kernel_time",
    ],
    use_merged_paradigm_description: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Generates a bar plot from the given benchmark DataFrame. and stores the result
    into a file if one is specified
    """
    df = benchmark_df.copy()
    # Map selected runtimes to DF columns
    runtime_map = {
        "wall_time": WALL_CLOCK_TIME,
        "cpu_time": CPU_TIME,
        "kernel_time": KERNEL_TIME,
    }
    runtimes = [runtime_map[r] for r in selected_runtime]

    # Column to use for framework (with optional version)
    name = PARADIGM
    if use_merged_paradigm_description:
        df[MERGED_PARADIGM_DESCRIPTION] = df.apply(
            lambda r: (
                r[PARADIGM]
                if r[DESCRIPTION] == ""
                else f"{r[PARADIGM]}[{r[DESCRIPTION]}]"
            ),
            axis=1,
        )
        name = MERGED_PARADIGM_DESCRIPTION

    # Prepare figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # One barplot per runtime stacked horizontally using dodge via hue
    # - x: framework (variants appear as separate categories)
    # - hue: problem size -> different colors
    # - col/linestyle not used; we overlay multiple runtime plots with slight offset
    palette = "tab20"
    frameworks_order = sorted(df[name].unique())
    size_order = sorted(df[PROBLEM_SIZE].unique())

    # To avoid overlapping when plotting multiple runtimes, create an x-position jitter per runtime
    # Compute a numeric x-position per category using category codes
    cat_codes = pd.Categorical(
        df[name], categories=frameworks_order, ordered=True
    ).codes
    # Base positions per row
    base_x = pd.Series(cat_codes, index=df.index)

    # Offset values centered around 0
    n_rt = max(1, len(runtimes))
    offsets = {rt: (i - (n_rt - 1) / 2) * 0.25 for i, rt in enumerate(runtimes)}

    # Plot each runtime as a separate barplot with dodged bars for sizes
    for rt in runtimes:
        df_rt = df.copy()
        df_rt["_xpos"] = base_x + offsets[rt]
        # Use seaborn barplot with estimator=mean for multiple entries per group
        sns.barplot(
            x="_xpos",
            y=rt,
            hue=PROBLEM_SIZE,
            hue_order=size_order,
            data=df_rt,
            ax=ax,
            palette=palette,
            ci=None,
            estimator="mean",
            dodge=True,
        )

    # Fix x-ticks to show framework labels centered (without offset)
    ax.set_xticks(range(len(frameworks_order)))
    ax.set_xticklabels(frameworks_order, rotation=20, ha="right")

    # Labels and title
    y_label = (
        " / ".join(selected_runtime)
        .replace("wall_time", "Wall Clock Time")
        .replace("cpu_time", "CPU Time")
        .replace("kernel_time", "Kernel Time")
    )
    ax.set_ylabel(f"{y_label} [{time_unit}]")
    ax.set_xlabel("Framework")
    # If Name exists, use first for title context
    title_name = (
        benchmark_df["Name"].iloc[0]
        if "Name" in benchmark_df.columns and not benchmark_df.empty
        else ""
    )
    ax.set_title(
        f"Runtime vs. Framework for {title_name}"
        if title_name
        else "Runtime vs. Framework"
    )

    # Merge duplicate legends (created per runtime overlay)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        # Keep only the last unique sequence for Problem Size
        unique = {}
        for h, l in zip(handles, labels):
            if l != "_xpos":  # seaborn artifact
                unique[l] = h
        ax.legend(unique.values(), unique.keys(), title="Problem Size")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path.with_name(save_path.stem + "_bar" + save_path.suffix), dpi=300
        )
        logger.info(
            f"Saved results to file {save_path.with_name(save_path.stem + '_bar' + save_path.suffix)}"
        )
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility to facilitate the orchestrated execution of all benchmarking targets and "
        "the data consolidation into a single output csv file. "
    )
    parser.add_argument(
        "-s",
        "--skip-benchmark",
        action="store_false",
        default=True,
        help="If set, the regex and path are used to directly search for the report files instead of benchmarking targets",
    )
    parser.add_argument(
        "-p",
        "--path",
        nargs="+",
        type=Path,
        default=[Path.cwd()],
        help="Path to search for files",
    )
    parser.add_argument(
        "-r",
        "--regex",
        nargs="+",
        type=str,
        required=True,
        help="Regex Pattern for files to search",
    )
    default_output = datetime.now().strftime("%Y-%m-%d_%H-%M_Benchmark_Result")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(default_output),
        help="Output file name for the csv and plots",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity level (Enable Debug & Trace Logs). Can be stacked.",
    )

    # Chart Arguments
    parser.add_argument(
        "-c",
        "--chart",
        type=str,
        default="none",
        help="Chart type to use for the plot. By default, no chart is generated.",
        choices=["none", "line", "bar"],
    )
    parser.add_argument(
        "--chart-time-unit",
        type=str,
        default="nanoseconds",
        help="Time unit for the plots",
    )
    parser.add_argument(
        "--chart-time-choice",
        type=str,
        default="wall_time",
        help="Time to use for the plot",
        choices=["wall_time", "cpu_time", "kernel_time"],
    )
    parser.add_argument(
        "--chart-merge-descriptor",
        action="store_true",
        default=False,
        help="Merge the Paradigm and Description into a single column for the plot",
    )

    args = parser.parse_args()

    # Set loguru log level to INFO, DEBUG, TRACE depending on the verbosity level
    logger.remove()
    logger.add(sys.stdout, level=["INFO", "DEBUG", "TRACE"][args.verbose])

    # Run the benchmark pipeline
    files = find_files(args.path, args.regex)
    if args.benchmark_reports:
        files = run_benchmarks(files)
    df = load_reports(files)
    df.to_csv(args.output.with_suffix(".csv"), index=False)
    match args.chart:
        case "line":
            plot_linechart(
                df,
                args.output.with_suffix(".pdf"),
                args.chart_time_unit,
                args.chart_time_choice,
                args.chart_merge_descriptor,
            )
        case "bar":
            plot_barchart(
                df,
                args.output.with_suffix(".pdf"),
                args.chart_time_unit,
                [args.chart_time_choice],
                args.chart_merge_descriptor,
            )
