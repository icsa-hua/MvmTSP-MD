from __future__ import annotations

from dummy_app.tools.logger import logger

import csv
import json
import pulp
import os
import time
import tracemalloc
import uuid

from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil


class Metrics:
    def __init__(self, base_dir: Optional[str] = None, verbose: bool = False) -> None:
        self.verbose = verbose
        self.base_dir = Path(base_dir or Path(os.getcwd()) / "assets" / "results" / "metrics")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.summary_file = self.base_dir / "run_summary.csv"
        self.reset()

    def reset(self) -> None:
        self.run_id = ""
        self.run_context: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_time: float = 0.0
        self.memory_usage: float = 0.0
        self.variables: Optional[int] = None
        self.constraints: Optional[int] = None
        self.cluster_records: List[Dict[str, Any]] = []
        self.coverage_records: List[Dict[str, Any]] = []
        self.latest_run_report: Dict[str, Any] = {}

    def start_run(self, run_context: Optional[Dict[str, Any]] = None, run_id: Optional[str] = None) -> str:
        self.reset()
        self.run_id = str(run_id or uuid.uuid4())
        self.run_context = dict(run_context or {})
        self.start_performance_timer()
        return self.run_id

    def start_tracemalloc(self) -> None:
        tracemalloc.start()

    def start_performance_timer(self) -> None:
        self.start_time = time.time()

    def end_performance_timer(self) -> float:
        self.end_time = time.time()
        if self.start_time is None or self.end_time is None:
            logger.error("Performance timer was not properly started or stopped.")
            self.elapsed_time = 0.0
        else:
            self.elapsed_time = self.end_time - self.start_time
        logger.info(f"Elapsed time: {self.elapsed_time} seconds")
        return self.elapsed_time

    def get_memory_usage(self) -> float:
        process = psutil.Process(os.getpid())
        self.memory_usage = process.memory_info().rss / (1024 ** 2)
        return self.memory_usage

    def record_cluster_result(self, cluster_result: Dict[str, Any]) -> None:
        self.cluster_records.append(dict(cluster_result))

    def record_coverage_result(self, coverage_result: Dict[str, Any]) -> None:
        self.coverage_records.append(dict(coverage_result))

    def build_run_report(
        self,
        summary: Dict[str, Any],
        cluster_results: Dict[str, Any],
        solve_status_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        report = {
            "run_id": self.run_id or str(uuid.uuid4()),
            "context": dict(self.run_context),
            "summary": dict(summary),
            "clusters": cluster_results,
            "cluster_records": list(self.cluster_records),
            "coverage_records": list(self.coverage_records),
            "solve_status_history": list(solve_status_history),
            "generated_at_epoch": time.time(),
        }
        self.latest_run_report = report
        return report


    def persist_run_report(self, report: Dict[str, Any]) -> Path:
        report_path = self.base_dir / f"{report['run_id']}.json"
        serializable_report = {str(k): v for k, v in report.items()}
        try: 
            with report_path.open("w", encoding="utf-8") as handle:
                json.dump(serializable_report, handle, indent=2, default=self._json_default, sort_keys=True)
        except Exception as e: 
            import pdb;pdb.set_trace()

        self._append_summary_row(report)
        return report_path


    def display_memory_usage(self) -> None:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")
        if not top_stats:
            logger.info("No tracemalloc statistics are available.")
            return
        logger.debug("Memory allocation snapshot:")
        logger.debug(f"The top memory-consuming variable: {top_stats[0]}")
        logger.debug(f"Total allocated memory: {top_stats[0].size / (1024 ** 2)} MB")
        logger.info(f"Total allocated memory: {top_stats[0].size / (1024 ** 2)} MB")


    def _append_summary_row(self, report: Dict[str, Any]) -> None:
        row = {
            "run_id": report["run_id"],
            **{f"context_{key}": value for key, value in report.get("context", {}).items()},
            **report.get("summary", {}),
            "cluster_records_count": len(report.get("cluster_records", [])),
            "coverage_records_count": len(report.get("coverage_records", [])),
        }
        file_exists = self.summary_file.exists()
        with self.summary_file.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow({key: self._stringify_value(value) for key, value in row.items()})


    @staticmethod
    def _stringify_value(value: Any) -> Any:

        if isinstance(value, (dict, list, tuple, set)):
            return json.dumps(value, default=Metrics._json_default, sort_keys=True)
        return value


    @staticmethod
    def _json_default(value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, pulp.pulp.LpVariable): 
            return value.value()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
