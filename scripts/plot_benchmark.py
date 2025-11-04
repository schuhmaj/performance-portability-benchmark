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


def find_files(search_dir: list[Path] | Path, pattern: str) -> list[Path]:
    """Searches for files given a search directory and a glob pattern.
    Args:
        search_dir: Directory/ Directories to search for files.
        pattern: Pattern to match files.

    Returns:
        A list of file paths that match the given pattern.
    """
    cmake_targets = []
    logger.info(f"Recursive searching for files in {search_dir} according to pattern: {pattern}")
    for directory in search_dir:
        for file in directory.rglob(pattern):
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
            subprocess.run(f"{target} --benchmark_out={output_file.name}", shell=True, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, )
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
    for file in report_files:
        logger.debug(f"Loading report file {file}")
        with open(file, "r") as file:
            json_data = json.load(file)
        df = pd.DataFrame(json_data["benchmarks"])
        # Assuming df is your DataFrame
        # 1. Drop rows where `iterations` is NaN
        df = df.dropna(subset=["iterations"])

        # 2. Select specific columns
        df = df[["name", "iterations", "real_time", "cpu_time", "time_unit", "kernel_time"]]

        # 3. Split the `name` column into `type`, `framework`, `precision`, and `size`
        # Assuming the format is consistent as 'VecAdd-Float-Kokkos-Version/1000'
        # Separate by '/' first, then '-'
        name_size_split = df["name"].str.split("/", expand=True)
        details_split = name_size_split[0].str.split("-", expand=True)


        # Create new columns from the splits
        df["Problem Size"] = name_size_split[1].astype(int)
        df["Name"] = details_split[0].astype(str)
        df["Precision"] = details_split[1].str.lower()
        df["Framework"] = details_split[2].astype(str)
        if len(details_split.columns) > 3:
            df["Version"] = details_split[3].fillna("").astype(str)
            df["Framework[Version]"] = df.apply(
                lambda r: r["Framework"] if r["Version"] == "" else f'{r["Framework"]}[{r["Version"]}]',
                axis=1
            )
        else:
            df["Version"] = ""
            df["Framework[Version]"] = df["Framework"]


        df.rename(
            columns={
                "real_time": "Wall Clock Time",
                "cpu_time": "CPU Time",
                "time_unit": "Time Unit",
                "iterations": "Iterations",
                "kernel_time" : "Kernel Time"
            },
            inplace=True
        )

        # Rearrange the columns as needed
        df = df[["Name", "Framework", "Precision", "Problem Size", "Iterations", "Wall Clock Time", "CPU Time", "Kernel Time", "Time Unit", "Version", "Framework[Version]"]]
        data.append(df)
    logger.info(f"Loaded {len(data)} report json files")
    return pd.concat(data)


def plot_linechart(benchmark_df: pd.DataFrame, save_path: Path | None = None, time_unit: str = "nanoseconds", selected_runtime: Literal["kernel_time", "wall_time", "cpu_time"] = "wall_time") -> tuple[plt.Figure, plt.Axes]:
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
    runtime = {"wall_time": "Wall Clock Time", "cpu_time": "CPU Time", "kernel_time": "Kernel Time"}[selected_runtime]
    name = "Framework[Version]"
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x="Problem Size", y=runtime, hue=name, hue_order=benchmark_df[name].unique().sort(), data=benchmark_df,
                 marker="o", ax=ax, palette="tab20", style="Precision")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(f"{runtime} [{time_unit}]")
    ax.set_xlabel("Problem Size [1]")
    ax.set_title(f'Runtime in {time_unit} vs. Problem Size $N$ for {benchmark_df["Name"].unique()[0]}')
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        logger.info(f"Saved results to file {save_path}")
    return fig, ax

def plot_barchart(benchmark_df: pd.DataFrame, save_path: Path | None = None, time_unit: str = "nanoseconds", selected_runtime: list[Literal["kernel_time", "wall_time", "cpu_time"]] = ["wall_time", "kernel_time"]) -> tuple[plt.Figure, plt.Axes]:
    """Generates a bar plot from the given benchmark DataFrame. and stores the result
    into a file if one is specified
    """
    # Map selected runtimes to DF columns
    runtime_map = {"wall_time": "Wall Clock Time", "cpu_time": "CPU Time", "kernel_time": "Kernel Time"}
    runtimes = [runtime_map[r] for r in selected_runtime]

    # Column to use for framework (with optional version)
    name_col = "Framework[Version]" if "Framework[Version]" in benchmark_df.columns else "Framework"

    # Prepare figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # One barplot per runtime stacked horizontally using dodge via hue
    # - x: framework (variants appear as separate categories)
    # - hue: problem size -> different colors
    # - col/linestyle not used; we overlay multiple runtime plots with slight offset
    palette = "tab20"
    frameworks_order = sorted(benchmark_df[name_col].unique())
    size_order = sorted(benchmark_df["Problem Size"].unique())

    # To avoid overlapping when plotting multiple runtimes, create an x-position jitter per runtime
    # Compute a numeric x-position per category using category codes
    cat_codes = pd.Categorical(benchmark_df[name_col], categories=frameworks_order, ordered=True).codes
    # Base positions per row
    base_x = pd.Series(cat_codes, index=benchmark_df.index)

    # Offset values centered around 0
    n_rt = max(1, len(runtimes))
    offsets = {rt: (i - (n_rt - 1) / 2) * 0.25 for i, rt in enumerate(runtimes)}

    # Plot each runtime as a separate barplot with dodged bars for sizes
    for rt in runtimes:
        df_rt = benchmark_df.copy()
        df_rt["_xpos"] = base_x + offsets[rt]
        # Use seaborn barplot with estimator=mean for multiple entries per group
        sns.barplot(
            x="_xpos",
            y=rt,
            hue="Problem Size",
            hue_order=size_order,
            data=df_rt,
            ax=ax,
            palette=palette,
            ci=None,
            estimator="mean",
            dodge=True
        )

    # Fix x-ticks to show framework labels centered (without offset)
    ax.set_xticks(range(len(frameworks_order)))
    ax.set_xticklabels(frameworks_order, rotation=20, ha="right")

    # Labels and title
    y_label = " / ".join(selected_runtime).replace("wall_time", "Wall Clock Time").replace("cpu_time", "CPU Time").replace("kernel_time", "Kernel Time")
    ax.set_ylabel(f"{y_label} [{time_unit}]")
    ax.set_xlabel("Framework")
    # If Name exists, use first for title context
    title_name = benchmark_df["Name"].iloc[0] if "Name" in benchmark_df.columns and not benchmark_df.empty else ""
    ax.set_title(f'Runtime vs. Framework for {title_name}' if title_name else "Runtime vs. Framework")

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
        fig.savefig(save_path.with_name(save_path.stem + "_bar" + save_path.suffix), dpi=300)
        logger.info(f"Saved results to file {save_path.with_name(save_path.stem + '_bar' + save_path.suffix)}")
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking Command Line Interface")
    parser.add_argument("--report", action="store_false", default=True,
        help="If set, the regex and path are used to directly search for the report files instead of benchmarking targets", )
    parser.add_argument("-p", "--path", nargs='+', type=Path, default=[Path.cwd()], help="Path to search for files")
    parser.add_argument("-r", "--regex", type=str, required=True, help="Regex Pattern for files to search")
    parser.add_argument("-u", "--time-unit",type=str, default="nanoseconds", help="Time unit for the plots",)
    default_output = datetime.now().strftime("%Y-%m-%d_%H-%M_Benchmark_Result.pdf")
    parser.add_argument("-o", "--output", type=Path, default=Path(default_output),
        help="Output file name for the plots", )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Verbosity level (Enable Debug & Trace Logs)")
    parser.add_argument("-t", "--time", type=str, default="wall_time", help="Time to use for the plot", choices=["wall_time", "cpu_time", "kernel_time"])
    parser.add_argument("-c","--chart", type=str, default="line", help="Chart type to use for the plot", choices=["line", "bar"])

    args = parser.parse_args()

    # Set loguru log level to INFO, DEBUG, TRACE depending on the verbosity level
    logger.remove()
    logger.add(sys.stdout, level=["INFO", "DEBUG", "TRACE"][args.verbose])

    # Run the benchmark pipeline
    files = find_files(args.path, args.regex)
    if args.report:
        files = run_benchmarks(files)
    df = load_reports(files)
    match args.chart:
        case "line":
            plot_linechart(df, args.output, args.time_unit, args.time)
        case "bar":
            plot_barchart(df, args.output, args.time_unit, [args.time])
    df.to_csv(args.output.with_suffix(".csv"), index=False)
