import unittest 
import pulp as pl
from dummy_app.models.builder import MVMTSPBuilder 

class TestMVMTSPBuilder(unittest.TestCase):
    def setUp(self):
        self.mvmtsp = MVMTSPBuilder(agents=[1, 2], max_battery=1000, time_frame=range(1, 11))
        self.V = [0, 1, 2, 3, 4]

    def test_createProblem_model_initialization(self):
        self.mvmtsp.createProblem(self.V)
        self.assertIsInstance(self.mvmtsp.model, pl.LpProblem)
        self.assertEqual(self.mvmtsp.model.name, "ConstrainedMvmTSP")

    def test_createProblem_x_variable(self):
        self.mvmtsp.createProblem(self.V)  
        for i in self.V:
            for j in self.V:
                for k in self.mvmtsp.agents:
                    self.assertIsInstance(self.mvmtsp.x[i, j, k], pl.LpVariable)
                    self.assertEqual(self.mvmtsp.x[i, j, k].cat, pl.LpBinary)


    def test_createProblem_u_variable(self):
        self.mvmtsp.createProblem(self.V)
        for i in self.V:
            for k in self.mvmtsp.agents:
                self.assertIsInstance(self.mvmtsp.u[i, k], pl.LpVariable)
                self.assertEqual(self.mvmtsp.u[i, k].cat, pl.LpInteger)
                self.assertEqual(self.mvmtsp.u[i, k].lowBound, 0)
                self.assertEqual(self.mvmtsp.u[i, k].upBound, len(self.V) - 1)


    def test_createProblem_t_variable(self):
        self.mvmtsp.createProblem(self.V)
        for i in self.V:
            for j in self.V:
                for k in self.mvmtsp.agents:
                    for ts in self.mvmtsp.TimeFrame:
                        self.assertIsInstance(self.mvmtsp.t[i, j, k, ts], pl.LpVariable)
                        self.assertEqual(self.mvmtsp.t[i, j, k, ts].cat, pl.LpBinary)


    def test_createProblem_e_variable(self):
        self.mvmtsp.createProblem(self.V)
        for i in self.V:
            for k in self.mvmtsp.agents:
                self.assertIsInstance(self.mvmtsp.e[i, k], pl.LpVariable)
                self.assertEqual(self.mvmtsp.e[i, k].cat, pl.LpContinuous)
                self.assertEqual(self.mvmtsp.e[i, k].lowBound, 0)
                self.assertEqual(self.mvmtsp.e[i, k].upBound, self.mvmtsp.max_battery)


    def test_createProblem_z_variable(self):
        self.mvmtsp.createProblem(self.V)
        for i in self.V:
            for k in self.mvmtsp.agents:
                self.assertIsInstance(self.mvmtsp.z[i, k], pl.LpVariable)
                self.assertEqual(self.mvmtsp.z[i, k].cat, pl.LpBinary)
                self.assertEqual(self.mvmtsp.z[i, k].lowBound, 0)
                self.assertEqual(self.mvmtsp.z[i, k].upBound, 1)


if __name__ == '__main__':
    unittest.main()