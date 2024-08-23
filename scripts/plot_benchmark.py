import os
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from loguru import logger

from pathlib import Path


def find_cmake_targets(search_dir: Path, pattern: str) -> list[Path]:
    cmake_targets = []
    logger.info(f"Recursive searching for executable targets in {search_dir} according to pattern: {pattern}")
    for file in search_dir.rglob(pattern):
        if file.is_file():
            cmake_targets.append(file.resolve())
    logger.info(f"Found {len(cmake_targets)} cmake targets")
    return cmake_targets


def run_benchmarks(executable_targets: list[Path]) -> list[Path]:
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
    data = []
    logger.info("Loading all report json files")
    for file in report_files:
        with open(file, "r") as file:
            json_data = json.load(file)
        df = pd.DataFrame(json_data["benchmarks"])
        # Assuming df is your DataFrame
        # 1. Drop rows where `iterations` is NaN
        df = df.dropna(subset=['iterations'])

        # 2. Select specific columns
        df = df[['name', 'iterations', 'real_time', 'cpu_time', 'time_unit']]

        # 3. Split the `name` column into `type`, `framework`, `precision`, and `size`
        # Assuming the format is consistent as 'VecAdd-Kokkos-Float/1000'
        # Separate by '/' first, then '-'
        name_split = df['name'].str.split('/', expand=True)
        details_split = name_split[0].str.split('-', expand=True)

        # Create new columns from the splits
        df['Problem'] = details_split[0]
        df['Framework'] = details_split[1].str.lower()
        df['Data Type'] = details_split[2].str.lower()
        df['Problem Size'] = name_split[1].astype(int)

        df.rename(
            columns={
                "real_time": "Runtime",
                "cpu_time": "CPU Time",
                "time_unit": "Time Unit",
                "iterations": "Iterations",
            },
            inplace=True
        )

        # Rearrange the columns as needed
        df = df[['Problem', 'Framework', 'Data Type', 'Problem Size', 'Iterations', 'Runtime', 'CPU Time', 'Time Unit']]
        data.append(df)
    logger.info(f"Loaded {len(data)} report json files")
    return pd.concat(data)


def plot_benchmarks(df: pd.DataFrame, save_path: Path = Path("./results.pdf")) -> tuple[plt.Figure, plt.Axes]:
    df = df[df['Data Type'] == 'float']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='Problem Size', y='Runtime', hue='Framework', data=df, marker='o', ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Runtime [ns]')
    ax.set_xlabel('Problem Size [1]')
    ax.set_title(f'Runtime in nanoseconds vs. Problem Size $N$ for {df["Problem"].unique()[0]}')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    logger.info(f"Saved results to file {save_path}")
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking Command Line Interface")
    parser.add_argument('-p', '--path', type=Path, default=Path.cwd(), help="Path to search for targets")
    parser.add_argument('-r', '--regex', type=str, required=True, help="Regex Pattern for targets to search")
    parser.add_argument('-o', '--output', type=Path, default=Path("result.pdf"), help="Output file name for the plots")

    args = parser.parse_args()

    # Use the arguments from the CLI
    search_path = args.path
    pattern = args.regex
    output_file = args.output

    # Run the benchmark pipeline
    targets = find_cmake_targets(search_path, pattern)
    files = run_benchmarks(targets)
    df = load_reports(files)
    plot_benchmarks(df, output_file)
    df.to_csv(output_file.with_suffix(".csv"), index=False)
