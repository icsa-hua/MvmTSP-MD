from __future__ import annotations

import unittest
from types import SimpleNamespace

from dummy_app.models.heuristics.common import list_service_nodes
from dummy_app.models.milp.warm_start import apply_warm_start_to_model


class WarmStartNodeTests(unittest.TestCase):
    def test_list_service_nodes_excludes_original_bridge_nodes(self) -> None:
        cluster = SimpleNamespace(
            depot_id=100,
            nodes_dict={0: 100, 1: 200, 2: 300, 3: 400, 4: 401},
            virtual_nodes={400: 300, 401: 300},
        )

        self.assertEqual(list_service_nodes(cluster), [200, 400, 401])

    def test_apply_warm_start_skips_unknown_runtime_nodes(self) -> None:
        cluster = SimpleNamespace(
            employed_agents=[1],
            depot_id=100,
            nodes_dict={0: 100, 1: 200, 2: 400},
            warm_start_summary={"strategy": "ga", "available": True},
            warm_start_solution={"task_sequences": {1: [99]}},
        )
        builder = SimpleNamespace(coverage_time=5.0, max_battery=100.0)

        result = apply_warm_start_to_model(cluster, builder)

        self.assertTrue(result["requested"])
        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "warm_start_unknown_node_99")


if __name__ == "__main__":
    unittest.main()
