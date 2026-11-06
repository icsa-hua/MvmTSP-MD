import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mvmtsp-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mvmtsp-xdg-cache")

from dummy_app.visualization.gantt import build_cluster_windows, build_gantt_segments, gantt_segments_to_frame


def test_depot_wait_splits_recorded_recharge_from_standby() -> None:
    plan_with_nodes = {
        1: [
            (0, 1, 0),
            (1, 1, 1),
            (1, 2, 2),
            (2, 2, 3),
            (2, 0, 4),
            (0, 0, 5),
            (0, 0, 6),
            (0, 0, 7),
            (0, 0, 8),
            (0, 0, 9),
        ],
        2: [
            (0, 3, 0),
            (3, 3, 1),
            (3, 0, 2),
            (0, 0, 3),
            (0, 0, 4),
            (0, 0, 5),
            (0, 0, 6),
            (0, 0, 7),
            (0, 0, 8),
            (0, 0, 9),
        ],
    }
    cluster_results = [
        {
            "cluster_id": 0,
            "agent_paths": {1: plan_with_nodes[1], 2: plan_with_nodes[2]},
            "agent_metrics": {
                1: {
                    "scheduled_start_time": 0,
                    "scheduled_finish_time": 5,
                    "next_available_time": 8,
                    "recharge_steps": 3,
                },
                2: {
                    "scheduled_start_time": 0,
                    "scheduled_finish_time": 3,
                    "next_available_time": 4,
                    "recharge_steps": 1,
                },
            },
        }
    ]

    schedule = gantt_segments_to_frame(build_gantt_segments(plan_with_nodes, cluster_results))

    recharge = schedule[schedule["segment_type"] == "recharge"].set_index("agent_id")
    standby = schedule[schedule["segment_type"] == "standby"].set_index("agent_id")

    assert recharge.loc["1", "start_step"] == 5
    assert recharge.loc["1", "end_step"] == 7
    assert recharge.loc["1", "duration"] == 3

    assert recharge.loc["2", "start_step"] == 3
    assert recharge.loc["2", "end_step"] == 3
    assert recharge.loc["2", "duration"] == 1
    assert standby.loc["2", "start_step"] == 4
    assert standby.loc["2", "end_step"] == 7

    assert build_cluster_windows(cluster_results, plan_with_nodes=plan_with_nodes) == [
        {"cluster_id": 0, "start_step": 0, "end_step": 7}
    ]


def test_gantt_trims_mutated_cluster_paths_and_infers_final_depot() -> None:
    plan_with_nodes = {
        1: [
            (5, 5, 0),
            (5, 9, 1),
            (9, 9, 2),
            (9, 9, 3),
            (9, 2, 4),
            (2, 9, 5),
            (9, 9, 6),
            (9, 9, 7),
        ]
    }
    cluster_results = [
        {
            "cluster_id": 0,
            # This mimics the old aliasing bug: the first cluster result was
            # mutated to contain the later flattened path too.
            "agent_paths": {1: plan_with_nodes[1]},
            "agent_metrics": {
                1: {
                    "scheduled_start_time": 0,
                    "scheduled_finish_time": 2,
                    "next_available_time": 4,
                    "recharge_steps": 2,
                }
            },
        },
        {
            "cluster_id": 1,
            "agent_paths": {1: [(9, 2, 0), (2, 9, 1), (9, 9, 2), (9, 9, 3)]},
            "agent_metrics": {
                1: {
                    "scheduled_start_time": 4,
                    "scheduled_finish_time": 6,
                    "next_available_time": 8,
                    "recharge_steps": 2,
                }
            },
        },
    ]

    schedule = gantt_segments_to_frame(build_gantt_segments(plan_with_nodes, cluster_results))

    assert schedule[
        (schedule["cluster_id"] == 0)
        & (schedule["segment_type"] == "service")
        & (schedule["label"] == "Service @ N9")
        & (schedule["start_step"] >= 2)
    ].empty

    recharge = schedule[schedule["segment_type"] == "recharge"].set_index("cluster_id")
    assert recharge.loc[0, "label"] == "Recharge @ N9"
    assert recharge.loc[0, "start_step"] == 2
    assert recharge.loc[0, "end_step"] == 3
    assert recharge.loc[1, "label"] == "Recharge @ N9"
    assert recharge.loc[1, "start_step"] == 6
    assert recharge.loc[1, "end_step"] == 7
