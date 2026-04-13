from __future__ import annotations

import argparse
import re
from pathlib import Path

PATTERN = re.compile(r"hidden_states\.shape\s*=\s*paddle\.Size\((\[[^\]]+\])\)")


def extract_shapes(log_path: Path) -> list[str]:
    shapes: list[str] = []
    seen: set[str] = set()
    with log_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            match = PATTERN.search(line)
            if not match:
                continue

            shape = match.group(1)
            if shape in seen:
                continue

            seen.add(shape)
            shapes.append(shape)
    return shapes


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="从 workerlog 中提取 hidden_states.shape 并写入 shape.txt"
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        default=script_dir / "workerlog.0",
        type=Path,
        help="日志文件路径，默认读取当前目录下的 workerlog.0",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default=script_dir / "shape.txt",
        type=Path,
        help="输出文件路径，默认写入当前目录下的 shape.txt",
    )
    args = parser.parse_args()

    shapes = extract_shapes(args.log_file)
    args.output_file.write_text("\n".join(shapes), encoding="utf-8")
    print(f"已提取 {len(shapes)} 条 shape 到 {args.output_file}")


if __name__ == "__main__":
    main()
