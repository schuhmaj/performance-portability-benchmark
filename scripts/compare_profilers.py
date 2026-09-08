#!/usr/bin/env python3
"""Compare the LIKWID and Nsight Compute profiling runs of the same binaries.

Both backends of ``ppbcc profile`` write the same columns, but they do not
measure the same unit of work: ``ncu`` reports one row per kernel launch, LIKWID
one row per marked region. The comparison is therefore made per *executable*,
with the ncu rows summed over the kernels that fall inside the marked region --
which for these benchmarks is every kernel the implementation launches.

Usage:
    python scripts/compare_profilers.py NCU_CSV LIKWID_CSV [-o OUTPUT_CSV]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

EXECUTABLE = "Executable"
PARADIGM = "Paradigm"
PROBLEM = "Benchmark Problem"
FLOP = "FLOP"
TRAFFIC = "Memory Traffic"
INTENSITY = "Arithmetic Intensity"
DURATION = "Duration"
PERFORMANCE = "Performance"

#: Kernels the runtimes launch once at start-up. They compute nothing and are
#: outside the marked region, so they must not enter the ncu aggregate either.
SETUP_KERNELS = ("init_lock_arrays", "query_cuda_kernel_arch")


def load(path: Path, exclude_setup: bool) -> pd.DataFrame:
    """Sum a profiling CSV per executable."""
    frame = pd.read_csv(path)
    if exclude_setup:
        pattern = "|".join(SETUP_KERNELS)
        frame = frame[~frame["Kernel"].astype(str).str.contains(pattern, na=False)]
    grouped = frame.groupby(EXECUTABLE, as_index=False).agg(
        {
            PROBLEM: "first",
            PARADIGM: "first",
            FLOP: "sum",
            TRAFFIC: "sum",
            DURATION: "sum",
        }
    )
    grouped[INTENSITY] = grouped[FLOP] / grouped[TRAFFIC]
    grouped[PERFORMANCE] = grouped[FLOP] / grouped[DURATION]
    return grouped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ncu", type=Path, help="CSV from --profiler ncu")
    parser.add_argument("likwid", type=Path, help="CSV from --profiler likwid")
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args(argv)

    ncu = load(args.ncu, exclude_setup=True)
    likwid = load(args.likwid, exclude_setup=False)
    merged = ncu.merge(likwid, on=EXECUTABLE, suffixes=(" ncu", " likwid"), how="inner")
    if merged.empty:
        print("No executable was profiled by both backends.", file=sys.stderr)
        return 1

    for column in (FLOP, TRAFFIC, INTENSITY, PERFORMANCE, DURATION):
        merged[f"{column} ratio"] = (
            merged[f"{column} likwid"] / merged[f"{column} ncu"]
        )

    columns = [EXECUTABLE, f"{PARADIGM} ncu"] + [
        f"{column}{suffix}"
        for column in (FLOP, TRAFFIC, INTENSITY, PERFORMANCE, DURATION)
        for suffix in (" ncu", " likwid", " ratio")
    ]
    table = merged[columns].sort_values(EXECUTABLE)

    with pd.option_context("display.width", 200, "display.max_columns", 40):
        summary = table[
            [EXECUTABLE, f"{FLOP} ratio", f"{TRAFFIC} ratio",
             f"{INTENSITY} ratio", f"{PERFORMANCE} ratio", f"{DURATION} ratio"]
        ]
        print(summary.to_string(index=False, float_format=lambda value: f"{value:7.3f}"))
        print("\nA ratio of 1.0 means the two backends agree.")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
