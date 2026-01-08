#!python3
import argparse
import sys
import json
import glob
from pathlib import Path

import pandas as pd

from loguru import logger


def load_files(file_path: Path) -> list[Path]:
    target_files = []
    logger.info(f"Recursive searching for JSON files in {file_path} ")
    for directory in file_path:
        for file in directory.rglob("*.json"):
            if file.is_file():
                target_files.append(file.resolve())
                logger.debug(f"Append file {file}")
    logger.info(f"Found {len(target_files)} files")
    return target_files


def create_dataframe(report_files: list[Path]) -> pd.DataFrame:
    """Creates a DataFrame from the given JSON files."""
    data = []
    logger.info("Loading all report json files")
    for file in report_files:
        logger.debug(f"Loading report file {file}")
        with open(file, "r") as f:
            json_data = json.load(f)
        df = pd.DataFrame(json_data["benchmarks"])
        # Assuming df is your DataFrame
        # 1. Drop rows where `iterations` is NaN
        df = df.dropna(subset=["iterations"])

        df["Name"] = "PolyhedralGravity"
        df["Precision"] = "float"
        df["Framework"] = str(file.stem[11:-7])
        framework_map = {
            "acpp": "AdaptiveCpp",
            "acc": "OpenACC",
            "cpp": "CPP",
            "kokkos": "Kokkos",
            "opencl": "OpenCL",
            "vulkan": "Vulkan",
            "slang_cuda": "Slang-Cuda",
            "slang_vulkan": "Slang-Vulkan",
            "omp": "OpenMP",
            "cuda": "Cuda",
        }
        # Apply mapping (unmapped values remain as-is with fillna)
        df["Framework"] = df["Framework"].map(framework_map).fillna(df["Framework"])
        df["Framework[Version]"] = df["Framework"]

        for col in [
            "kernel_time",
            "position_update_reset",
            "velocity_update",
            "force_update",
            "Version",
        ]:
            if col not in df.columns:
                df[col] = None

        df.rename(
            columns={
                "NumFaces": "Problem Size",
                "real_time": "Wall Clock Time",
                "cpu_time": "CPU Time",
                "time_unit": "Time Unit",
                "iterations": "Iterations",
                "kernel_time": "Kernel Time",
                "position_update_reset": "Position Update Time",
                "velocity_update": "Velocity Update Time",
                "force_update": "Force Update Time",
            },
            inplace=True,
        )

        if "Problem Size" in df.columns:
            df["Problem Size"] = pd.to_numeric(
                df["Problem Size"], errors="coerce"
            ).astype("Int64")

        # Rearrange the columns as needed
        df = df[
            [
                "Name",
                "Framework",
                "Precision",
                "Problem Size",
                "Iterations",
                "Wall Clock Time",
                "CPU Time",
                "Kernel Time",
                "Position Update Time",
                "Velocity Update Time",
                "Force Update Time",
                "Time Unit",
                "Version",
                "Framework[Version]",
            ]
        ]
        data.append(df)
    logger.info(f"Loaded {len(data)} report json files")
    return pd.concat(data)


def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Selects only the entries where column `Problem Size` is 14744 or 255932.
    Only selects one row for each `Framework[Version]`
    """
    return data[
        (data["Problem Size"] == 14744) | (data["Problem Size"] == 255932)
    ].drop_duplicates(subset=["Framework[Version]", "Problem Size"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking Command Line Interface")
    parser.add_argument(
        "--path",
        nargs="+",
        type=Path,
        default=[Path.cwd()],
        help="Path to search for files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity level (Enable Debug & Trace Logs)",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level=["INFO", "DEBUG", "TRACE"][args.verbose])

    report_files = load_files(args.path)
    df = create_dataframe(report_files)
    df = filter_data(df)
    df.to_csv("polyhedral_benchmark.csv", index=False)
