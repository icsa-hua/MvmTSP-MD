from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "results" / "experiment_D_objective_staging.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "table_experiment_D_objective_staging.tex"

SINGLE_STAGE = "single_stage_milp"
TWO_STAGE = "two_stage_milp"

BASELINE_METRICS = {
    "coverage_ratio": "coverage",
    "total_distance": "distance",
    "total_energy": "energy",
    "total_travel_time": "travel_time",
    "workload_balance_distance_std": "distance_balance",
    "workload_balance_energy_std": "energy_balance",
    "secondary_objective_value": "two_stage_gap",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an IEEE-style LaTeX table for Experiment D objective staging results."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _to_float(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    stripped = str(raw_value).strip()
    if not stripped:
        return None
    return float(stripped)


def _mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return float(mean(valid))


def _delta_percent(two_stage_value: float | None, single_stage_value: float | None) -> float | None:
    if two_stage_value is None or single_stage_value is None:
        return None
    if abs(float(single_stage_value)) <= 1e-12:
        return 0.0 if abs(float(two_stage_value)) <= 1e-12 else None
    return float((float(two_stage_value) - float(single_stage_value)) * 100.0 / float(single_stage_value))


def _format_value(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "---"
    return f"{float(value):.{digits}f}"


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_by_area(rows: list[dict[str, str]]) -> list[dict[str, float | int | None]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        area = int(row["node_count"])
        formulation = str(row["formulation_type"])
        grouped[(area, formulation)].append(row)

    summaries: list[dict[str, float | int | None]] = []
    areas = sorted({area for area, _ in grouped.keys()})

    for area in areas:
        single_rows = grouped.get((area, SINGLE_STAGE), [])
        two_rows = grouped.get((area, TWO_STAGE), [])
        if not single_rows or not two_rows:
            continue

        single_runtime = _mean([_to_float(row.get("runtime_sec")) for row in single_rows])
        two_runtime = _mean([_to_float(row.get("runtime_sec")) for row in two_rows])
        two_coverage_ratio = _mean([_to_float(row.get("coverage_ratio")) for row in two_rows])

        deltas: dict[str, float | None] = {}
        for metric_name, output_name in BASELINE_METRICS.items():
            single_value = _mean([_to_float(row.get(metric_name)) for row in single_rows])
            two_value = _mean([_to_float(row.get(metric_name)) for row in two_rows])
            deltas[output_name] = _delta_percent(two_value, single_value)

        summaries.append(
            {
                "area": area,
                "single_runtime": single_runtime,
                "two_runtime": two_runtime,
                "delta_coverage": deltas["coverage"],
                "delta_distance": deltas["distance"],
                "delta_energy": deltas["energy"],
                "delta_travel_time": deltas["travel_time"],
                "delta_distance_balance": deltas["distance_balance"],
                "delta_energy_balance": deltas["energy_balance"],
                # "two-stage gap" is interpreted as the delta on the stage-2 objective
                # (secondary objective / makespan) relative to the single-stage solution.
                "delta_two_stage_gap": deltas["two_stage_gap"],
                "coverage_ratio": two_coverage_ratio,
            }
        )

    return summaries


def build_latex_table(summaries: list[dict[str, float | int | None]]) -> str:
    body_lines = []
    for row in summaries:
        body_lines.append(
            "    {area} & {single_runtime} & {two_runtime} & {delta_coverage} & {delta_distance} & "
            "{delta_energy} & {delta_travel_time} & {delta_distance_balance} & {delta_energy_balance} & "
            "{delta_two_stage_gap}  & {coverage_ratio} \\\\".format(
                area=int(row["area"]),
                single_runtime=_format_value(row["single_runtime"], digits=2),
                two_runtime=_format_value(row["two_runtime"], digits=2),
                delta_coverage=_format_value(row["delta_coverage"], digits=2),
                delta_distance=_format_value(row["delta_distance"], digits=2),
                delta_energy=_format_value(row["delta_energy"], digits=2),
                delta_travel_time=_format_value(row["delta_travel_time"], digits=2),
                delta_distance_balance=_format_value(row["delta_distance_balance"], digits=2),
                delta_energy_balance=_format_value(row["delta_energy_balance"], digits=2),
                delta_two_stage_gap=_format_value(row["delta_two_stage_gap"], digits=2),
                coverage_ratio=_format_value(row["coverage_ratio"], digits=3),
            )
        )

    body = "\n".join(body_lines)
    return f"""% IEEE Transactions -- Experiment D: Objective staging comparison
% Auto-generated by diagrams/generate_experiment_D_table.py from results/experiment_D_objective_staging.csv

\\begin{{table*}}[t]
  \\centering
  \\caption{{Objective Staging Performance Summary (Experiment D, seed=42, \\textit{{medium}} coverage profile).
           Results are averaged across $k \\in \\{{3,4,6,8\\}}$ for each area count. Percentage deltas are computed as
           $\\Delta X(\\%) = 100(X_{{2\\text{{S}}}} - X_{{1\\text{{S}}}})/X_{{1\\text{{S}}}}$; negative values indicate that
           the two-stage formulation reduced the metric. $\\Delta$Gap is computed from the stage-2 objective
           (secondary objective / makespan), Opt.~Gap is the reported two-stage MILP optimality gap, and
           $\\rho_{{\\mathrm{{cov}}}}$ is the achieved two-stage coverage ratio.}}
  \\label{{tab:exp_d_objective_staging}}
  \\renewcommand{{\\arraystretch}}{{1.15}}
  \\setlength{{\\tabcolsep}}{{4pt}}
  \\begin{{tabular}}{{r | rr | rrrrrr | rrr}}
    \\hline\\hline
    & \\multicolumn{{2}}{{c|}}{{\\textbf{{Runtime (s)}}}} &
      \\multicolumn{{6}}{{c|}}{{\\textbf{{Percentage Deltas (Two-Stage vs Single-Stage)}}}} &
      \\multicolumn{{3}}{{c}}{{\\textbf{{Two-Stage Outcome}}}} \\\\
    \\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-9}}\\cmidrule(lr){{10-12}}
    \\textbf{{Areas}} & Single & Two-Stage & $\\Delta$Cov. & $\\Delta$Dist. & $\\Delta$Energy &
      $\\Delta$Time & $\\Delta$Bal.$_D$ & $\\Delta$Bal.$_E$ & $\\Delta$Gap & Opt.~Gap & $\\rho_{{\\mathrm{{cov}}}}$ \\\\
    \\hline
{body}
    \\hline\\hline
  \\end{{tabular}}
\\end{{table*}}
"""


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    summaries = aggregate_by_area(rows)
    latex = build_latex_table(summaries)
    args.output.write_text(latex, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
