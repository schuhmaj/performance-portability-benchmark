import os
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def find_cmake_targets(search_dir: Path) -> list[Path]:
    cmake_targets = []
    for file in search_dir.rglob("vec_*"):
        if file.is_file():
            cmake_targets.append(file.resolve())
    return cmake_targets


def run_benchmarks(executable_targets: list[Path]) -> None:
    for target in executable_targets:
        output_file = f"{target.name}_report.json"
        cmd = f"{target} --benchmark_out={output_file}"
        try:
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to execute {target}: {e}")


def collect_reports(search_dir: Path) -> pd.DataFrame:
    data = []
    for report_file in search_dir.rglob("*report.json"):
        if report_file.is_file():
            with open(report_file, "r") as file:
                report = json.load(file)
                for benchmark in report["benchmarks"]:
                    benchmark["filename"] = str(report_file)
                    data.append(benchmark)
    return pd.DataFrame(data)


def plot_benchmarks(data: pd.DataFrame):
    plt.figure(figsize=(14, 7))
    for name, group in df.groupby("filename"):
        plt.plot(group["name"], group["real_time"], label=name, marker="o")

    plt.xlabel("Benchmark")
    plt.ylabel("Runtime (ns)")
    plt.title("Benchmark Runtime Measurements")
    plt.legend(loc="best")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("benchmark_plot.png")
    plt.show()


if __name__ == "__main__":
    directory = Path("./cmake-build-release")  # the directory to search for targets
    targets = find_cmake_targets(directory)
    # run_benchmarks(targets)
    df = collect_reports(Path(os.getcwd()))
    if not df.empty:
        plot_benchmarks(df)
    else:
        print("No benchmark reports found.")
