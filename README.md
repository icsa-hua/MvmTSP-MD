# Many-Visits Multi-Agent TSP for Energy- and Communication-Aware UAV Swarm Mission Scheduling in SAR Operations

An auditable Python framework for planning Search-and-Rescue (SAR) UAV-swarm missions in which UAVs act as temporary Non-Terrestrial Network (NTN) base stations. It formulates mission scheduling as an energy- and communication-aware **Many-Visits Multi-Agent Traveling Salesperson Problem (MVMTSP)** and combines realistic map, mobility, energy, and coverage simulation with exact and heuristic solvers.

This repository accompanies the accepted IEEE Transactions on Vehicular Technology paper, [*Many-Visits Multi-Agent TSP for Energy- and Communication-Aware UAV Swarm Mission Scheduling in SAR Operations*](https://doi.org/10.1109/TVT.2026.3750670).

![License](https://img.shields.io/badge/license-GPL-blue.svg)
![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue.svg)

## Highlights

- Cooperative coverage (**COOP-UC**), where the swarm jointly covers the map, and individual coverage (**INDI-UC**), where every UAV completes a full traversal.
- MILP scheduling with battery feasibility, repeated visits through Virtual Node Installation (VNI), service-time synchronization, workload balancing, recharging, and collision-aware coordination.
- Energy, coverage/data-rate, time, and staged objective strategies.
- MILP backends for GLPK, CBC, Gurobi, and CPLEX; MTZ, iterative DFJ, and flow-based subtour elimination.
- Heuristic and metaheuristic baselines: global Greedy Nearest Neighbor, static-partition Greedy NN, Genetic Algorithm, and ALNS; these can also provide MILP warm starts.
- Reproducible experiment drivers for solver, initializer, subtour, objective, scalability, stress, VNI-growth, and COOP-versus-INDI studies.
- Persisted JSON/CSV artifacts and optional trajectory playback/animation.

## Results from the paper

The reported experiments use a fixed seed (42), realistic simulation inputs, and a unified Python interface. Results are simulation-based; exact values depend on solver availability, hardware, limits, and generated instance parameters.

- On the method-comparison study, the MILP achieved the lowest mean route energy (797.1 Wh) and distance (8.28 km). The heuristic methods were consistently feasible and much faster, at the cost of approximately 5.1–5.3% more energy and 4.1–6.6% more distance.
- Exact energy optimization is valuable for tractable instances and certification, but it becomes solver-sensitive as instances grow. In the paper, the MILP found a feasible solution in 92.2% of comparison runs and certified 89 of 180 runs as optimal. Communication-coverage optimization scaled more readily.
- Across the COOP-UC versus INDI-UC experiments, cooperative scheduling reduced energy, distance, and mission time by about 54–56%, while more than doubling nodes served per unit of energy and time. INDI-UC can yield a larger aggregate data rate through redundant full-map traversals; COOP-UC delivers better data-rate efficiency.
- Warm starts and alternate subtour formulations can improve individual runs, but neither removes the inherent scaling boundary of exact MVMTSP scheduling.

For methodology, metrics, limits, and complete tables, see the supplied paper PDF and the DOI above.

## Requirements

- Python 3.9 or newer
- GNU GLPK (`glpsol`) for the default open-source MILP workflow
- Python dependencies in [`requirements.txt`](requirements.txt)

GLPK installation:

```sh
# Debian/Ubuntu
sudo apt-get install glpk-utils

# macOS (Homebrew)
brew install glpk
```

Gurobi and CPLEX are optional commercial backends. Install and license them separately before selecting either backend. CBC is available when supported by the local PuLP installation.

## Setup

```sh
git clone https://github.com/icsa-hua/MvmTSP-MD.git
cd MvmTSP-MD
python3 -m pip install -e .
```

The default runtime settings are in [`dummy_app/program_config.py`](dummy_app/program_config.py). They include a Gurobi default; for an open-source installation, pass `--solver_backend glpk` (or update that setting) when running a scenario.

## Run a scenario

Run the default simulation with GLPK:

```sh
python3 execution_script.py --solver_backend glpk
```

For a small cooperative, energy-focused example:

```sh
python3 execution_script.py \
  --scenario cooperative \
  --objective energy \
  --num_areas 21 \
  --num_agents 3 \
  --num_users 15 \
  --env urban \
  --solver_backend glpk \
  --subtour_mode mtz \
  --warm_start_mode auto \
  --seed 42
```

| Option | Values |
| --- | --- |
| `--scenario` | `cooperative`, `individual` |
| `--objective` | `energy`, `coverage`, `sum_of_times`, `pareto` |
| `--solver_backend` | `glpk`, `cbc`, `gurobi`, `cplex` |
| `--subtour_mode` | `mtz`, `dfj_iter`, `flow` |
| `--warm_start_mode` | `auto`, `none`, `ga`, `alns`, `greedy_nn`, `greedy_partition_nn` |
| `--mode` | `solve`, `animate`, `solve_and_animate` |

Use `python3 execution_script.py --help` for all options, including solver/run time limits, animation options, validation, and dataset/RL workflows. Instance generation is stochastic unless a seed is supplied.

### Artifacts and animation

Each solve persists a replayable run under `assets/runs/<run-id>/`:

- `instance.json`, `run_request.json`, and `run_result.json`
- `paths.csv` with per-agent path steps
- `cluster_metrics.csv` with status, objective, bound, gap, and timing data

Use `--mode solve_and_animate --save_animation` to solve and save an MP4, or replay a persisted artifact:

```sh
python3 execution_script.py \
  --mode animate \
  --artifact_dir assets/runs/<run-id> \
  --save_animation
```

## Reproduce paper experiments

The scripts in [`reproduce/`](reproduce) write CSV outputs to `results/` by default. They use the experiment grids and limits in `dummy_app/program_config.py`; some studies can take hours and commercial-solvers scripts require the applicable local installation.

```sh
# Main comparison: MILP, GA, Greedy NN, static-partition Greedy NN, and ALNS
python3 reproduce/experiment_A_main_comparison.py

# MILP backend comparison
python3 reproduce/experiment_B_solver_comparison.py

# Warm starts, objective staging, sensitivity, scalability, stress, and VNI growth
python3 reproduce/experiment_C_warm_start.py
python3 reproduce/experiment_D_objective_staging.py
python3 reproduce/experiment_E_sensitivity.py
python3 reproduce/experiment_F_scalability.py
python3 reproduce/experiment_G_extreme_constraint_stress.py
python3 reproduce/experiment_H_vni_growth.py

# Objective and cooperative-versus-individual comparisons
python3 reproduce/experiment_I_objective_comparison.py
python3 reproduce/experiment_COOP_vs_INDI.py

# Subtour study and path visualizations
python3 reproduce/experiment_Subtour.py
python3 reproduce/experiment_J_path_objectives.py

# Create plots from available result CSVs
python3 -m dummy_app.experiments.plot_results
```

Each experiment accepts `--help`; most accept `--output` and `--append`. The scalability driver additionally exposes hard memory and wall-clock caps.

## Project layout

```text
dummy_app/
  designs/         Scenario, UAV, mobility, map, and configuration models
  models/          Simulation pipeline, energy/coverage models, MILP, heuristics, and RL
  pipeline/        Typed instance/run contracts and persistent artifacts
  experiments/     Metrics and result plotting
  visualization/   Playback and Gantt/trajectory visualizations
  tools/           Logging, solver reporting, and utility scripts
reproduce/         Paper experiment drivers and batch scripts
assets/            Generated run artifacts, logs, animations, and data (created at runtime)
results/           Experiment CSVs and figures (created at runtime)
```

## Docker

Docker provides a Linux environment with GLPK. From the repository root:

```sh
docker compose run --rm mvmtsp-app python3 execution_script.py --solver_backend glpk
```

The first run creates a named virtual environment and installs the package. Generated artifacts remain available on the host through the bind-mounted repository, primarily in `assets/runs/` (and in `results/` for experiment scripts).

## Citation

```bibtex
@article{georgiadis2026mvmtsp,
  title={Many-Visits Multi-Agent TSP for Energy- and Communication-Aware UAV Swarm Mission Scheduling in SAR Operations},
  author={Georgiadis, Dimitrios and Vamvakari, Malvina and Politi, Elena and Dimitrakopoulos, George},
  journal={IEEE Transactions on Vehicular Technology},
  year={2026},
  doi={10.1109/TVT.2026.3750670}
}
```

## License

This project is distributed under the GPL license. See the repository license information for details.
