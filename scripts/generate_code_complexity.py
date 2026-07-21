#!/usr/bin/env python3
"""Generate file-level and implementation-level code-complexity results.

The implementation manifest below is intentionally explicit.  In particular,
it separates implementations that share a directory and counts every Slang
shader once with its CUDA host and once with its Vulkan host.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RESULTS = ROOT / "results" / "code-complexity"
FILE_RESULTS = RESULTS / "code-complexity-files.csv"
AGGREGATE_RESULTS = RESULTS / "code-complexity.csv"

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


def files_below(relative: str, *, exclude_main: bool = True) -> tuple[str, ...]:
    """Return source files below a directory, relative to the repository root."""
    directory = ROOT / relative
    files = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if exclude_main and path.stem == "main":
            continue
        files.append(path.relative_to(ROOT).as_posix())
    return tuple(files)


def impl(
    problem: str,
    framework: str,
    dialect: str,
    *sources: str,
) -> Implementation:
    return Implementation(problem, framework, dialect, tuple(sources))


def implementation_manifest() -> tuple[Implementation, ...]:
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
        entries.append(impl("VecAdd", framework, dialect, *files_below(f"src/vectorAdditon/{directory}")))
    vector_cuda_header = "src/vectorAdditon/cuda/Implementations.cuh"
    entries.extend(
        (
            impl("VecAdd", "Cuda", "cuda", vector_cuda_header, "src/vectorAdditon/cuda/Impl_Cuda.cu"),
            impl("VecAdd", "Cublas", "cuda", vector_cuda_header, "src/vectorAdditon/cuda/Impl_Cublas.cu"),
            impl("VecAdd", "Cuda[Chunked]", "cuda", vector_cuda_header, "src/vectorAdditon/cuda/Impl_ChunkedCuda.cu"),
            impl("VecAdd", "Thrust", "cuda,thrust", vector_cuda_header, "src/vectorAdditon/cuda/Impl_Thrust.cu"),
            impl("VecAdd", "Stdpar[NVHPC]", "stdpar", "src/vectorAdditon/cuda/Impl_NvhpcStd.cpp"),
            impl(
                "VecAdd", "Slang-Cuda", "slang,cuda",
                "src/vectorAdditon/slang/Impl_SlangCuda.cu",
                "src/vectorAdditon/slang/VectorAdditionShader.slang",
            ),
            impl(
                "VecAdd", "Slang-Vulkan", "slang,vulkan",
                "src/vectorAdditon/slang/Impl_SlangVulkan.cpp",
                "src/vectorAdditon/slang/VectorAdditionShader.slang",
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
        sources = list(files_below(f"src/matrixMultiplication/{directory}"))
        if framework == "Boost":
            sources.append("src/matrixMultiplication/opencl/MatrixMultiplication.cl")
        entries.append(impl("MatrixMultiplication", framework, dialect, *sources))
    entries.extend(
        (
            impl("MatrixMultiplication", "AdaptiveCpp[Naive]", "sycl", "src/matrixMultiplication/acpp/Impl_AdaptiveCpp.cpp", "src/matrixMultiplication/acpp/Impl_AdaptiveCpp.h"),
            impl("MatrixMultiplication", "AdaptiveCpp[SharedMemory]", "sycl", "src/matrixMultiplication/acpp/Impl_AdaptiveCppShr.cpp", "src/matrixMultiplication/acpp/Impl_AdaptiveCppShr.h"),
            impl("MatrixMultiplication", "Cublas", "cuda", "src/matrixMultiplication/cuda/Impl_Cublas.cu", "src/matrixMultiplication/cuda/Impl_Cublas.cuh"),
            impl("MatrixMultiplication", "Cuda[Naive]", "cuda", "src/matrixMultiplication/cuda/Impl_CudaNaive.cu", "src/matrixMultiplication/cuda/Impl_CudaNaive.cuh"),
            impl("MatrixMultiplication", "Cuda[SharedMemory]", "cuda", "src/matrixMultiplication/cuda/Impl_Cuda.cu", "src/matrixMultiplication/cuda/Impl_Cuda.cuh"),
            impl("MatrixMultiplication", "Cuda[Buffer]", "cuda", "src/matrixMultiplication/cuda/Impl_CudaBuffer.cu", "src/matrixMultiplication/cuda/Impl_CudaBuffer.cuh"),
            impl("MatrixMultiplication", "Cuda[Tensor]", "cuda", "src/matrixMultiplication/cuda/Impl_CudaTensor.cu", "src/matrixMultiplication/cuda/Impl_CudaTensor.cuh"),
            impl("MatrixMultiplication", "OpenMP", "openmp", "src/matrixMultiplication/openmp/Impl_OpenMPDevice.cpp", "src/matrixMultiplication/openmp/Impl_OpenMPDevice.h"),
            impl("MatrixMultiplication", "OpenMP[Host]", "openmp", "src/matrixMultiplication/openmp/Impl_OpenMP.cpp", "src/matrixMultiplication/openmp/Impl_OpenMP.h"),
            impl(
                "MatrixMultiplication", "Vulkan", "vulkan,glsl",
                "src/matrixMultiplication/vulkan/Impl_Vulkan.cpp",
                "src/matrixMultiplication/vulkan/Impl_Vulkan.h",
                "src/matrixMultiplication/vulkan/MatrixMultiplicationShader.comp",
            ),
            impl(
                "MatrixMultiplication", "Vulkan[SharedMemory]", "vulkan,glsl",
                "src/matrixMultiplication/vulkan/Impl_Vulkan.cpp",
                "src/matrixMultiplication/vulkan/Impl_Vulkan.h",
                "src/matrixMultiplication/vulkan/MatrixMultiplicationShaderShr.comp",
            ),
            impl(
                "MatrixMultiplication", "Slang-Cuda", "slang,cuda",
                "src/matrixMultiplication/slang/Impl_SlangCuda.cu",
                "src/matrixMultiplication/slang/Impl_SlangCuda.cuh",
                "src/matrixMultiplication/slang/MatrixMultiplicationShader.slang",
            ),
            impl(
                "MatrixMultiplication", "Slang-Vulkan", "slang,vulkan",
                "src/matrixMultiplication/slang/Impl_SlangVulkan.cpp",
                "src/matrixMultiplication/slang/Impl_SlangVulkan.h",
                "src/matrixMultiplication/slang/MatrixMultiplicationShader.slang",
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
        sources = list(files_below(f"src/nBodySimulation/{directory}"))
        if framework == "Boost":
            sources.append("src/nBodySimulation/opencl/ForceKernel.cl")
        entries.append(impl("NBody", framework, dialect, *sources))
    kokkos_base = (
        "src/nBodySimulation/kokkos/Impl_Kokkos.cpp",
        "src/nBodySimulation/kokkos/Impl_Kokkos.h",
    )
    entries.append(impl("NBody", "Kokkos", "kokkos", *kokkos_base))
    entries.append(
        impl(
            "NBody", "Kokkos[Reduction]", "kokkos", *kokkos_base,
            "src/nBodySimulation/kokkos/Impl_KokkosReduction.cpp",
            "src/nBodySimulation/kokkos/Impl_KokkosReduction.h",
        )
    )
    for variant_dir, description in (
        ("naive", "Naive"),
        ("cell_lists", "LinkedCells"),
        ("verlet_lists", "VerletLists"),
    ):
        vulkan_host = files_below(f"src/nBodySimulation/vulkan/{variant_dir}")
        vulkan_shaders = tuple(path for path in vulkan_host if path.endswith(".comp"))
        vulkan_cpp = tuple(path for path in vulkan_host if not path.endswith(".comp"))
        slang_cuda = files_below(f"src/nBodySimulation/slang/{variant_dir}")
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
        ("CPP[128-bit]", "cpp", "cpp_128"),
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
        sources = list(files_below(f"src/polyhedralGravity/{directory}"))
        if framework == "Boost":
            sources.extend(files_below("src/polyhedralGravity/opencl/kernel"))
        entries.append(impl("PolyhedralGravity", framework, dialect, *sources))
    entries.extend(
        (
            impl(
                "PolyhedralGravity", "Slang-Cuda", "slang,cuda,thrust",
                "src/polyhedralGravity/slang/Impl_Slang_Cuda.cu",
                "src/polyhedralGravity/slang/wrapper_eval.cu",
                "src/polyhedralGravity/slang/shader/eval.slang",
            ),
            impl(
                "PolyhedralGravity", "Slang-Vulkan", "slang,vulkan",
                "src/polyhedralGravity/slang/Impl_Slang_Vulkan.cpp",
                "src/polyhedralGravity/slang/shader/eval.slang",
            ),
        )
    )
    return tuple(entries)


def source_index() -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Build exact-relative and basename indexes for local include resolution."""
    exact: dict[str, Path] = {}
    basename: dict[str, list[Path]] = {}
    for path in sorted(SRC.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        relative_src = path.relative_to(SRC).as_posix()
        exact[relative_src] = path
        basename.setdefault(path.name, []).append(path)
    return exact, basename


def local_dependencies(seed_sources: tuple[str, ...]) -> tuple[Path, ...]:
    """Resolve repository-local quoted includes and common implementation units."""
    exact, basename = source_index()
    pending = [(ROOT / relative).resolve() for relative in seed_sources]
    selected: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in selected:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Manifest source does not exist: {path.relative_to(ROOT)}")
        selected.add(path)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped.startswith("#include \""):
                continue
            include = stripped.removeprefix("#include \"").split('"', 1)[0]
            candidates = [path.parent / include, SRC / include]
            dependency = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if dependency is None:
                dependency = exact.get(include)
            if dependency is None and len(basename.get(Path(include).name, [])) == 1:
                dependency = basename[Path(include).name][0]
            if dependency is not None and dependency.suffix.lower() in SOURCE_SUFFIXES:
                pending.append(dependency.resolve())

        # Linked common utilities have a .cpp implementation beside the header.
        if path.suffix.lower() in {".h", ".hpp", ".cuh"} and SRC / "common" in path.parents:
            companion = path.with_suffix(".cpp")
            if companion.is_file():
                pending.append(companion.resolve())
    return tuple(sorted(selected))


def command_for(
    sources: tuple[Path, ...],
    output: Path,
    *,
    dialect: str | None = None,
    aggregate: bool = False,
) -> list[str]:
    command = [sys.executable, "-m", "code_complexity"]
    command.extend(path.relative_to(ROOT).as_posix() for path in sources)
    if dialect is not None:
        command.extend(("--dialect", dialect))
    if aggregate:
        command.append("--aggregate")
        command.extend(("--metrics", *AGGREGATE_METRICS))
    command.extend(("--output", output.relative_to(ROOT).as_posix() if output.is_relative_to(ROOT) else str(output)))
    return command


def run(command: list[str], *, dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def generate(dry_run: bool) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_sources = tuple(
        path for path in sorted(SRC.rglob("*"))
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
    )
    run(command_for((SRC,), FILE_RESULTS), dry_run=dry_run)

    aggregate_rows: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="code-complexity-") as temporary:
        temporary_dir = Path(temporary)
        for index, implementation in enumerate(implementation_manifest()):
            sources = local_dependencies(implementation.sources)
            output = temporary_dir / f"aggregate-{index:03d}.csv"
            run(
                command_for(sources, output, dialect=implementation.dialect, aggregate=True),
                dry_run=dry_run,
            )
            if dry_run:
                continue
            with output.open(newline="", encoding="utf-8") as handle:
                total = next(row for row in csv.DictReader(handle) if row["file"] == "TOTAL")
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
    with AGGREGATE_RESULTS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("Name", "Framework", "SLOC", "n1", "n2", "N1", "N2"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    with FILE_RESULTS.open(newline="", encoding="utf-8") as handle:
        file_rows = list(csv.DictReader(handle))
    if len(file_rows) != len(all_sources):
        raise RuntimeError(
            f"File-level report has {len(file_rows)} rows; expected {len(all_sources)} source files"
        )
    expected_pairs = {(entry.problem, entry.framework) for entry in implementation_manifest()}
    actual_pairs = {(row["Name"], row["Framework"]) for row in aggregate_rows}
    if len(expected_pairs) != len(aggregate_rows) or actual_pairs != expected_pairs:
        raise RuntimeError("Aggregate report is missing or duplicates an implementation row")

    print(f"Wrote {len(file_rows)} file rows to {FILE_RESULTS.relative_to(ROOT)}")
    print(f"Wrote {len(aggregate_rows)} implementation rows to {AGGREGATE_RESULTS.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print every python -m code_complexity command without executing it",
    )
    arguments = parser.parse_args()
    generate(arguments.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
