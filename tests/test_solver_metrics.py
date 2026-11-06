from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

from dummy_app.experiments.runner_utils import _extract_failure_metrics, extract_common_run_metrics
from dummy_app.models.milp.solver_adapter import (
    _infer_warm_start_acceptance_from_progress,
    _parse_cplex_progress,
    _parse_gurobi_progress,
    _parse_warm_start_acceptance_from_log,
    _resolve_effective_raw_status,
)


class SolverLogParsingTests(unittest.TestCase):
    def _write_log(self, contents: str) -> str:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "solver.log"
        path.write_text(textwrap.dedent(contents).strip() + "\n", encoding="utf-8")
        return str(path)

    def test_gurobi_time_limit_is_not_treated_as_optimal(self) -> None:
        log_path = self._write_log(
            """
            Found heuristic solution: objective 1703.0000000
            Root relaxation: time limit, 313 iterations, 0.00 seconds (0.01 work units)

                Nodes    |    Current Node    |     Objective Bounds      |     Work
             Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

                 0     0          -    0      1703.00000    0.00000   100%     -    0s

            Explored 1 nodes (313 simplex iterations) in 0.01 seconds (0.02 work units)
            Time limit reached
            Best objective 1.703000000000e+03, best bound 0.000000000000e+00, gap 100.0000%
            """
        )
        summary = _parse_gurobi_progress(log_path, 1703.0, "Optimal")

        self.assertFalse(summary["optimality_proven"])
        self.assertTrue(summary["feasible_solution_found"])
        self.assertEqual(summary["final_incumbent_value"], 1703.0)
        self.assertEqual(summary["final_best_bound"], 0.0)
        self.assertEqual(summary["final_relative_gap_percent"], 100.0)
        self.assertEqual(_resolve_effective_raw_status("Optimal", summary), "Feasible")

    def test_cplex_time_limit_is_not_treated_as_optimal(self) -> None:
        log_path = self._write_log(
            """
            Found incumbent of value 453.000000 after 0.00 sec. (4.50 ticks)

                    Nodes                                         Cuts/
               Node  Left     Objective  IInf  Best Integer    Best Bound    ItCnt     Gap

            *     0+    0                          453.0000        0.0000           100.00%
                  0     0  -1.00000e+37     0      453.0000        0.0000      324  100.00%

            MIP - Time limit exceeded, integer feasible:  Objective =  4.5300000000e+02
            Current MIP best bound =  0.0000000000e+00 (gap = 453, 100.00%)
            Solution time =    0.01 sec.  Iterations = 324  Nodes = 0
            """
        )
        summary = _parse_cplex_progress(log_path, 453.0, "Optimal")

        self.assertFalse(summary["optimality_proven"])
        self.assertTrue(summary["feasible_solution_found"])
        self.assertEqual(summary["final_incumbent_value"], 453.0)
        self.assertEqual(summary["final_best_bound"], 0.0)
        self.assertEqual(summary["final_relative_gap_percent"], 100.0)
        self.assertEqual(_resolve_effective_raw_status("Optimal", summary), "Feasible")

    def test_cplex_mip_start_acceptance_is_reported(self) -> None:
        log_path = self._write_log(
            """
            MIP start 'm1' defined initial solution with objective 237.2894638615.
            1 of 1 MIP starts provided solutions.
            Found incumbent of value 237.289464 after 0.00 sec. (1.23 ticks)
            MIP - Integer optimal solution:  Objective =  2.372894638615e+02
            Solution time =    0.01 sec.  Iterations = 15  Nodes = 0
            """
        )

        acceptance = _parse_warm_start_acceptance_from_log(log_path, "cplex")

        self.assertTrue(acceptance["accepted"])
        self.assertEqual(acceptance["source"], "solver_log_accepted")
        self.assertAlmostEqual(acceptance["objective_value"], 237.2894638615)

    def test_first_incumbent_match_can_infer_mip_start_acceptance(self) -> None:
        acceptance = _infer_warm_start_acceptance_from_progress(
            {
                "progress_events": [
                    {
                        "elapsed_time_seconds": 0.0,
                        "incumbent_value": 237.2894638614729,
                    }
                ]
            },
            237.2894638614729,
        )

        self.assertTrue(acceptance["accepted"])
        self.assertEqual(acceptance["source"], "first_incumbent_match")
        self.assertAlmostEqual(acceptance["objective_value"], 237.2894638614729)


class ExperimentMetricTests(unittest.TestCase):
    def test_gap_uses_visible_solver_objective_not_summary_cost(self) -> None:
        cluster_result = SimpleNamespace(
            incumbent_value=3.7132204003955276,
            best_bound=3.7132204003955276,
            elapsed_time_seconds=11.0,
            normalized_status="optimal",
            diagnostics={"first_feasible_time_seconds": 0.0, "explored_bnb_nodes": 0},
            agent_metrics={},
            agent_paths={},
        )
        run_result = SimpleNamespace(
            cluster_results=[cluster_result],
            summary={"memory_usage_mb": 244.625, "objective_value": 87.43997436184435},
            objective_value=3.7132204003955276,
            summary_objective_value=87.43997436184435,
            elapsed_time_seconds=11.554578065872192,
            normalized_status="optimal",
        )

        metrics = extract_common_run_metrics(run_result, {"target_nodes": []})

        self.assertEqual(metrics["objective_value"], 3.7132204003955276)
        self.assertEqual(metrics["summary_objective_value"], 87.43997436184435)
        self.assertEqual(metrics["best_bound"], 3.7132204003955276)
        self.assertEqual(metrics["optimality_gap_percent"], 0.0)

    def test_failure_metrics_classify_time_limit_without_solution(self) -> None:
        builder = SimpleNamespace(
            solve_status_history=[
                {
                    "time_limit_reached": True,
                    "incumbent_value": None,
                    "best_bound": 236.0,
                    "explored_bnb_nodes": 0,
                }
            ],
            metrics=SimpleNamespace(memory_usage=123.0),
        )

        metrics = _extract_failure_metrics(builder, "RuntimeError: Cluster 0 ended with status Not Solved")

        self.assertTrue(metrics["time_limit_no_solution"])
        self.assertFalse(metrics["time_limit_feasible"])
        self.assertFalse(metrics["model_build_error"])
        self.assertFalse(metrics["solver_error"])
        self.assertEqual(metrics["best_bound"], 236.0)
        self.assertIsNone(metrics["objective_value"])


if __name__ == "__main__":
    unittest.main()
