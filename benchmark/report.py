#!/usr/bin/env python3
import argparse
import csv
import io
import json
from pathlib import Path


PREFERRED_COLUMNS = (
    "benchmark",
    "op",
    "implementation",
    "device",
    "device_id",
    "dtype",
    "shape",
    "prompt_tokens",
    "output_tokens",
    "mean_ms",
    "median_ms",
    "p90_ms",
    "p99_ms",
    "ttft_mean_ms",
    "tpot_mean_ms",
    "decode_tokens_per_s",
    "repeat",
    "git_revision",
)


def load_records(path):
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"Invalid JSON on {path}:{line_number}: {error}") from error
    return records


def columns_for(records):
    available = {key for record in records for key in record}
    preferred = [column for column in PREFERRED_COLUMNS if column in available]
    return preferred + sorted(available - set(preferred))


def display(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return str(value)


def render_csv(records, columns):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        writer.writerow({column: display(record.get(column)) for column in columns})
    return stream.getvalue()


def render_markdown(records, columns):
    def cell(value):
        return display(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(record.get(column)) for column in columns) + " |"
        for record in records
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Convert benchmark JSONL to a report.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--format", choices=("csv", "markdown"), default="markdown")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--implementation", choices=("torch", "llaisys"), default=None)
    args = parser.parse_args()

    records = load_records(args.input)
    if args.implementation:
        records = [r for r in records if r.get("implementation") == args.implementation]
    if not records:
        raise SystemExit("No benchmark records matched")

    columns = columns_for(records)
    rendered = (
        render_csv(records, columns)
        if args.format == "csv"
        else render_markdown(records, columns)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
