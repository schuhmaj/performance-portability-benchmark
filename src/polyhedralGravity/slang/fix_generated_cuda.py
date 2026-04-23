from pathlib import Path
import sys


def main(target: Path):
    new_file = target.with_name(target.name[4:])
    replaced = target.read_text().replace(
        'extern "C" __constant__ GlobalParams_0', "__device__ GlobalParams_0"
    )

    replaced = replaced.replace("void computeMain()", f"void run_{new_file.stem}()")

    new_file.write_text(replaced)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
