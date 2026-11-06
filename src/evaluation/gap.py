from __future__ import annotations

from typing import Optional

from dummy_app.core.statuses import compute_absolute_gap, compute_relative_gap, compute_relative_gap_percent


def build_gap_summary(incumbent: Optional[float], best_bound: Optional[float]) -> dict[str, Optional[float]]:
    return {
        "incumbent_value": incumbent,
        "best_bound": best_bound,
        "absolute_gap": compute_absolute_gap(incumbent, best_bound),
        "relative_gap": compute_relative_gap(incumbent, best_bound),
        "relative_gap_percent": compute_relative_gap_percent(incumbent, best_bound),
    }
