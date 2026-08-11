#!/usr/bin/env python3
"""Generate file-level and implementation-level code-complexity results.

The implementation manifest below is intentionally explicit.  In particular,
it separates implementations that share a directory and counts every Slang
shader once with its CUDA host and once with its Vulkan host.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger
from ppbcc.code_complexity import AUTO_DIALECT_NAME, evaluate, save_csv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "src"
DEFAULT_OUTPUT = ROOT / "results" / "code-complexity"
FILE_RESULTS_NAME = "code-complexity-files.csv"
AGGREGATE_RESULTS_NAME = "code-complexity.csv"

#: Every manifest path is relative to the source folder passed via ``--source``.
PROBLEM_SOURCE_ROOTS = {
    "VecAdd": "vectorAdditon/",
    "MatrixMultiplication": "matrixMultiplication/",
    "NBody": "nBodySimulation/",
    "PolyhedralGravity": "polyhedralGravity/",
}

SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".c++",
    ".h", ".hh", ".hpp", ".hxx", ".h++", ".inl", ".inc", ".ipp",
    ".cu", ".cuh", ".hip", ".cl", ".comp", ".glsl", ".vert", ".frag",
    ".geom", ".tesc", ".tese", ".slang", ".hlsl", ".wgsl", ".metal",
}

AGGREGATE_METRICS = (
    "sloc",
    "distinct_operators",
    "distinct_operands",
    "total_operators",
    "total_operands",
)


@dataclass(frozen=True)
class Implementation:
    problem: str
    framework: str
    dialect: str
    sources: tuple[str, ...]


def files_below(source: Path, relative: str, *, exclude_main: bool = True) -> tuple[str, ...]:
    """Return source files below a directory, relative to the source folder."""
    directory = source / relative
    if not directory.is_dir():
        raise FileNotFoundError(f"Manifest directory does not exist: {directory}")
    files = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if exclude_main and path.stem == "main":
            continue
        files.append(path.relative_to(source).as_posix())
    return tuple(files)


def impl(
    problem: str,
    framework: str,
    dialect: str,
    *sources: str,
) -> Implementation:
    return Implementation(problem, framework, dialect, tuple(sources))


def implementation_manifest(source: Path) -> tuple[Implementation, ...]:
    """Describe every logical implementation and its implementation-owned files."""
    entries: list[Implementation] = []

    # Vector addition. CUDA contains implementations not currently registered
    # in its executable; they remain distinct source implementations.
    vector_single = (
        ("CPP", "cpp", "cpp"),
        ("AdaptiveCpp", "sycl", "acpp"),
        ("Alpaka", "alpaka", "alpaka"),
        ("Boost", "boost_compute", "boost"),
        ("Hip", "hip", "hip"),
        ("Kokkos", "kokkos", "kokkos"),
        ("Metal", "metal", "metal"),
        ("OpenACC", "openacc", "openacc"),
        ("OpenCL", "opencl", "opencl"),
        ("OpenMP", "openmp", "openmp"),
        ("RAJA", "raja", "raja"),
        ("Stdpar", "stdpar", "stdpar"),
        ("Vulkan", "vulkan,glsl", "vulkan"),
    )
    for framework, dialect, directory in vector_single:
        entries.append(impl("VecAdd", framework, dialect, *files_below(source, f"vectorAdditon/{directory}")))
    vector_cuda_header = "vectorAdditon/cuda/Implementations.cuh"
    entries.extend(
        (
            impl("VecAdd", "Cuda", "cuda", vector_cuda_header, "vectorAdditon/cuda/Impl_Cuda.cu"),
            impl("VecAdd", "Cublas", "cuda", vector_cuda_header, "vectorAdditon/cuda/Impl_Cublas.cu"),
            impl("VecAdd", "Cuda[Chunked]", "cuda", vector_cuda_header, "vectorAdditon/cuda/Impl_ChunkedCuda.cu"),
            impl("VecAdd", "Thrust", "cuda,thrust", vector_cuda_header, "vectorAdditon/cuda/Impl_Thrust.cu"),
            impl("VecAdd", "Stdpar[NVHPC]", "stdpar", "vectorAdditon/cuda/Impl_NvhpcStd.cpp"),
            impl(
                "VecAdd", "Slang-Cuda", "slang,cuda",
                "vectorAdditon/slang/Impl_SlangCuda.cu",
                "vectorAdditon/slang/VectorAdditionShader.slang",
            ),
            impl(
                "VecAdd", "Slang-Vulkan", "slang,vulkan",
                "vectorAdditon/slang/Impl_SlangVulkan.cpp",
                "vectorAdditon/slang/VectorAdditionShader.slang",
            ),
        )
    )

    # Matrix multiplication variants sharing AdaptiveCpp, CUDA, OpenMP, and
    # Vulkan directories are split so unrelated kernels do not inflate a row.
    matrix_single = (
        ("CPP", "cpp", "cpp"),
        ("Alpaka", "alpaka", "alpaka"),
        ("Boost", "boost_compute,opencl", "boost"),
        ("Hip", "hip", "hip"),
        ("Kokkos", "kokkos", "kokkos"),
        ("OpenACC", "openacc", "openacc"),
        ("OpenCL", "opencl", "opencl"),
        ("RAJA", "raja", "raja"),
        ("Stdpar", "stdpar", "stdpar"),
    )
    for framework, dialect, directory in matrix_single:
        sources = list(files_below(source, f"matrixMultiplication/{directory}"))
        if framework == "Boost":
            sources.append("matrixMultiplication/opencl/MatrixMultiplication.cl")
        entries.append(impl("MatrixMultiplication", framework, dialect, *sources))
    entries.extend(
        (
            impl("MatrixMultiplication", "AdaptiveCpp[Naive]", "sycl", "matrixMultiplication/acpp/Impl_AdaptiveCpp.cpp", "matrixMultiplication/acpp/Impl_AdaptiveCpp.h"),
            impl("MatrixMultiplication", "AdaptiveCpp[SharedMemory]", "sycl", "matrixMultiplication/acpp/Impl_AdaptiveCppShr.cpp", "matrixMultiplication/acpp/Impl_AdaptiveCppShr.h"),
            impl("MatrixMultiplication", "Cublas", "cuda", "matrixMultiplication/cuda/Impl_Cublas.cu", "matrixMultiplication/cuda/Impl_Cublas.cuh"),
            impl("MatrixMultiplication", "Cuda[Naive]", "cuda", "matrixMultiplication/cuda/Impl_CudaNaive.cu", "matrixMultiplication/cuda/Impl_CudaNaive.cuh"),
            impl("MatrixMultiplication", "Cuda[SharedMemory]", "cuda", "matrixMultiplication/cuda/Impl_Cuda.cu", "matrixMultiplication/cuda/Impl_Cuda.cuh"),
            impl("MatrixMultiplication", "Cuda[Buffer]", "cuda", "matrixMultiplication/cuda/Impl_CudaBuffer.cu", "matrixMultiplication/cuda/Impl_CudaBuffer.cuh"),
            impl("MatrixMultiplication", "Cuda[Tensor]", "cuda", "matrixMultiplication/cuda/Impl_CudaTensor.cu", "matrixMultiplication/cuda/Impl_CudaTensor.cuh"),
            impl("MatrixMultiplication", "OpenMP", "openmp", "matrixMultiplication/openmp/Impl_OpenMPDevice.cpp", "matrixMultiplication/openmp/Impl_OpenMPDevice.h"),
            impl("MatrixMultiplication", "OpenMP[Host]", "openmp", "matrixMultiplication/openmp/Impl_OpenMP.cpp", "matrixMultiplication/openmp/Impl_OpenMP.h"),
            impl(
                "MatrixMultiplication", "Vulkan", "vulkan,glsl",
                "matrixMultiplication/vulkan/Impl_Vulkan.cpp",
                "matrixMultiplication/vulkan/Impl_Vulkan.h",
                "matrixMultiplication/vulkan/MatrixMultiplicationShader.comp",
            ),
            impl(
                "MatrixMultiplication", "Vulkan[SharedMemory]", "vulkan,glsl",
                "matrixMultiplication/vulkan/Impl_Vulkan.cpp",
                "matrixMultiplication/vulkan/Impl_Vulkan.h",
                "matrixMultiplication/vulkan/MatrixMultiplicationShaderShr.comp",
            ),
            impl(
                "MatrixMultiplication", "Slang-Cuda", "slang,cuda",
                "matrixMultiplication/slang/Impl_SlangCuda.cu",
                "matrixMultiplication/slang/Impl_SlangCuda.cuh",
                "matrixMultiplication/slang/MatrixMultiplicationShader.slang",
            ),
            impl(
                "MatrixMultiplication", "Slang-Vulkan", "slang,vulkan",
                "matrixMultiplication/slang/Impl_SlangVulkan.cpp",
                "matrixMultiplication/slang/Impl_SlangVulkan.h",
                "matrixMultiplication/slang/MatrixMultiplicationShader.slang",
            ),
        )
    )

    # NBody: Kokkos Reduction derives from the regular Kokkos implementation,
    # so its aggregate includes both base and derived source files.
    nbody_single = (
        ("CPP", "cpp", "cpp"),
        ("AdaptiveCpp", "sycl", "acpp"),
        ("Alpaka", "alpaka", "alpaka"),
        ("Boost", "boost_compute,opencl", "boost"),
        ("Cuda", "cuda", "cuda"),
        ("Hip", "hip", "hip"),
        ("OpenACC", "openacc", "openacc"),
        ("OpenCL", "opencl", "opencl"),
        ("OpenMP", "openmp", "openmp"),
        ("RAJA", "raja", "raja"),
        ("Stdpar", "stdpar", "stdpar"),
    )
    for framework, dialect, directory in nbody_single:
        sources = list(files_below(source, f"nBodySimulation/{directory}"))
        if framework == "Boost":
            sources.append("nBodySimulation/opencl/ForceKernel.cl")
        entries.append(impl("NBody", framework, dialect, *sources))
    kokkos_base = (
        "nBodySimulation/kokkos/Impl_Kokkos.cpp",
        "nBodySimulation/kokkos/Impl_Kokkos.h",
    )
    entries.append(impl("NBody", "Kokkos", "kokkos", *kokkos_base))
    entries.append(
        impl(
            "NBody", "Kokkos[Reduction]", "kokkos", *kokkos_base,
            "nBodySimulation/kokkos/Impl_KokkosReduction.cpp",
            "nBodySimulation/kokkos/Impl_KokkosReduction.h",
        )
    )
    for variant_dir, description in (
        ("naive", "Naive"),
        ("cell_lists", "LinkedCells"),
        ("verlet_lists", "VerletLists"),
    ):
        vulkan_host = files_below(source, f"nBodySimulation/vulkan/{variant_dir}")
        vulkan_shaders = tuple(path for path in vulkan_host if path.endswith(".comp"))
        vulkan_cpp = tuple(path for path in vulkan_host if not path.endswith(".comp"))
        slang_cuda = files_below(source, f"nBodySimulation/slang/{variant_dir}")
        slang_shaders = tuple(path for path in slang_cuda if path.endswith(".slang"))
        slang_cuda_host = tuple(path for path in slang_cuda if not path.endswith(".slang"))
        entries.extend(
            (
                impl("NBody", f"Vulkan[{description}]", "vulkan,glsl", *vulkan_cpp, *vulkan_shaders),
                impl("NBody", f"Slang-Cuda[{description}]", "slang,cuda", *slang_cuda_host, *slang_shaders),
                impl("NBody", f"Slang-Vulkan[{description}]", "slang,vulkan", *vulkan_cpp, *slang_shaders),
            )
        )

    # Polyhedral gravity implementations share PolyhedralGravityDefinitions.h,
    # which is discovered through local includes. Boost reuses OpenCL kernels.
    poly_single = (
        ("CPP", "cpp", "cpp"),
        ("AdaptiveCpp", "sycl", "acpp"),
        ("Alpaka", "alpaka", "alpaka"),
        ("Boost", "boost_compute,opencl", "boost"),
        ("Cuda", "cuda,thrust", "cuda"),
        ("Hip", "hip", "hip"),
        ("Kokkos", "kokkos", "kokkos"),
        ("OpenACC", "openacc", "openacc"),
        ("OpenCL", "opencl", "opencl"),
        ("OpenMP", "openmp", "openmp"),
        ("RAJA", "raja", "raja"),
        ("Stdpar", "stdpar", "stdpar"),
        ("Vulkan", "vulkan,glsl", "vulkan"),
        ("WebGPU", "webgpu,wgsl", "webgpu"),
    )
    for framework, dialect, directory in poly_single:
        sources = list(files_below(source, f"polyhedralGravity/{directory}"))
        if framework == "Boost":
            sources.extend(files_below(source, "polyhedralGravity/opencl/kernel"))
        entries.append(impl("PolyhedralGravity", framework, dialect, *sources))
    entries.extend(
        (
            impl(
                "PolyhedralGravity", "Slang-Cuda", "slang,cuda,thrust",
                "polyhedralGravity/slang/Impl_Slang_Cuda.cu",
                "polyhedralGravity/slang/wrapper_eval.cu",
                "polyhedralGravity/slang/shader/eval.slang",
            ),
            impl(
                "PolyhedralGravity", "Slang-Vulkan", "slang,vulkan",
                "polyhedralGravity/slang/Impl_Slang_Vulkan.cpp",
                "polyhedralGravity/slang/shader/eval.slang",
            ),
        )
    )
    return tuple(entries)


def all_source_files(source: Path) -> tuple[Path, ...]:
    """Return every analysable source file below the source folder."""
    return tuple(
        path for path in sorted(source.rglob("*"))
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
    )


def source_index(source: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Build exact-relative and basename indexes for local include resolution."""
    exact: dict[str, Path] = {}
    basename: dict[str, list[Path]] = {}
    for path in all_source_files(source):
        exact[path.relative_to(source).as_posix()] = path
        basename.setdefault(path.name, []).append(path)
    return exact, basename


def local_dependencies(source: Path, seed_sources: tuple[str, ...]) -> tuple[Path, ...]:
    """Resolve repository-local quoted includes and common implementation units."""
    exact, basename = source_index(source)
    pending = [(source / relative).resolve() for relative in seed_sources]
    selected: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in selected:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Manifest source does not exist: {path}")
        selected.add(path)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped.startswith("#include \""):
                continue
            include = stripped.removeprefix("#include \"").split('"', 1)[0]
            candidates = [path.parent / include, source / include]
            dependency = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if dependency is None:
                dependency = exact.get(include)
            if dependency is None and len(basename.get(Path(include).name, [])) == 1:
                dependency = basename[Path(include).name][0]
            if dependency is not None and dependency.suffix.lower() in SOURCE_SUFFIXES:
                pending.append(dependency.resolve())

        # Linked common utilities have a .cpp implementation beside the header.
        if path.suffix.lower() in {".h", ".hpp", ".cuh"} and source / "common" in path.parents:
            companion = path.with_suffix(".cpp")
            if companion.is_file():
                pending.append(companion.resolve())
    return tuple(sorted(selected))


def validate_dependencies(
    source: Path,
    implementation: Implementation,
    sources: tuple[Path, ...],
) -> None:
    """Reject empty manifest entries and dependencies on a different problem."""
    if not sources:
        raise RuntimeError(
            f"{implementation.problem}/{implementation.framework} resolves to no "
            f"source file below {source}; the manifest entry is stale"
        )
    expected_root = PROBLEM_SOURCE_ROOTS[implementation.problem]
    problem_roots = tuple(PROBLEM_SOURCE_ROOTS.values())
    relative_sources = tuple(path.relative_to(source).as_posix() for path in sources)
    foreign_sources = [
        path
        for path in relative_sources
        if path.startswith(problem_roots) and not path.startswith(expected_root)
    ]
    if foreign_sources:
        formatted = ", ".join(foreign_sources)
        raise RuntimeError(
            f"{implementation.problem}/{implementation.framework} includes "
            f"source from another benchmark problem: {formatted}"
        )


def reported_path(source: Path, path: Path) -> str:
    """Render a path the way it should appear in the report, e.g. ``src/a/b.cpp``."""
    parent = source.parent
    return path.relative_to(parent).as_posix() if path.is_relative_to(parent) else path.as_posix()


def describe(
    source: Path,
    sources: tuple[Path, ...],
    *,
    dialect: str | None = None,
    aggregate: bool = False,
) -> str:
    """Render the analysis about to run the way the CLI would spell it."""
    parts = ["code-complexity", *(reported_path(source, path) for path in sources)]
    if dialect is not None:
        parts.extend(("--dialect", dialect))
    if aggregate:
        parts.append("--aggregate")
        parts.extend(("--metrics", *AGGREGATE_METRICS))
    return " ".join(parts)


def analyze(
    source: Path,
    sources: tuple[Path, ...],
    *,
    dialect: str | None = None,
    aggregate: bool = False,
    dry_run: bool,
) -> pd.DataFrame | None:
    logger.info("{}", describe(source, sources, dialect=dialect, aggregate=aggregate))
    if dry_run:
        return None
    return evaluate(
        sources=list(sources),
        language_dialect=AUTO_DIALECT_NAME if dialect is None else dialect,
        metrics=list(AGGREGATE_METRICS) if aggregate else None,
        aggregate=aggregate,
    )


def generate(source: Path, output: Path, *, dry_run: bool) -> None:
    """Write both reports for ``source`` into the ``output`` directory."""
    file_results = output / FILE_RESULTS_NAME
    aggregate_results = output / AGGREGATE_RESULTS_NAME
    manifest = implementation_manifest(source)
    all_sources = all_source_files(source)
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)

    file_frame = analyze(source, (source,), dry_run=dry_run)
    if file_frame is not None:
        # ``evaluate`` records the paths it was handed; report them relative to
        # the parent of the source folder so the CSV stays machine independent.
        file_frame["file"] = [reported_path(source, Path(name)) for name in file_frame["file"]]
        save_csv(file_frame, file_results)

    aggregate_rows: list[dict[str, str]] = []
    for implementation in manifest:
        sources = local_dependencies(source, implementation.sources)
        validate_dependencies(source, implementation, sources)
        frame = analyze(source, sources, dialect=implementation.dialect, aggregate=True, dry_run=dry_run)
        if frame is None:
            continue
        total = frame.loc[frame["file"] == "TOTAL"].iloc[0]
        aggregate_rows.append(
            {
                "Name": implementation.problem,
                "Framework": implementation.framework,
                "SLOC": total["sloc"],
                "n1": total["distinct_operators"],
                "n2": total["distinct_operands"],
                "N1": total["total_operators"],
                "N2": total["total_operands"],
            }
        )

    if dry_run:
        return
    with aggregate_results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("Name", "Framework", "SLOC", "n1", "n2", "N1", "N2"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    if len(file_frame) != len(all_sources):
        raise RuntimeError(
            f"File-level report has {len(file_frame)} rows; expected {len(all_sources)} source files"
        )
    expected_pairs = {(entry.problem, entry.framework) for entry in manifest}
    actual_pairs = {(row["Name"], row["Framework"]) for row in aggregate_rows}
    if len(expected_pairs) != len(aggregate_rows) or actual_pairs != expected_pairs:
        raise RuntimeError("Aggregate report is missing or duplicates an implementation row")

    logger.success("Wrote {} file rows to {}", len(file_frame), file_results)
    logger.success("Wrote {} implementation rows to {}", len(aggregate_rows), aggregate_results)


def configure_logging(verbosity: int) -> None:
    """Configure loguru; ppbcc's own per-file chatter needs at least one ``-v``."""
    logger.remove()
    logger.add(
        sys.stderr,
        level={0: "INFO", 1: "DEBUG"}.get(verbosity, "TRACE"),
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
    )
    if verbosity == 0:
        logger.disable("ppbcc")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-s",
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        metavar="DIR",
        help=f"source folder holding the implementations (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="DIR",
        help=(
            f"directory for {AGGREGATE_RESULTS_NAME} and {FILE_RESULTS_NAME} "
            f"(default: {DEFAULT_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="log every code-complexity analysis without running it",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase verbosity (-v: ppbcc debug output, -vv: trace)",
    )
    arguments = parser.parse_args()
    configure_logging(arguments.verbose)

    source = arguments.source.resolve()
    if not source.is_dir():
        logger.error("Source folder does not exist: {}", source)
        return 1
    try:
        generate(source, arguments.output.resolve(), dry_run=arguments.dry_run)
    except (FileNotFoundError, KeyError, RuntimeError) as error:
        logger.error("{}", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
