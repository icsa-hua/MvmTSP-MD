#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NUM_AGENTS="${NUM_AGENTS:-3}"
NUM_AREAS="${NUM_AREAS:-21}"
NUM_USERS="${NUM_USERS:-1}"
ENV_TYPE="${ENV_TYPE:-urban}"
OBJECTIVE="${OBJECTIVE:-energy}"
STAGE_SOLUTION="${STAGE_SOLUTION:-1}"
TRIALS="${TRIALS:-1}"
ENABLE_GA="${ENABLE_GA:-yes}"
SOLVER_BACKEND="${SOLVER_BACKEND:-glpk}"
SOLVER_TIME_LIMIT_SECONDS="${SOLVER_TIME_LIMIT_SECONDS:-30}"
OUTPUT_DIR="${OUTPUT_DIR:-assets/results/solver_comparison/$(date +%Y%m%d_%H%M%S)}"

print_help() {
    cat <<EOF
Usage: ./compare_solver_baselines.sh [options]

Runs the four baseline configurations:
  1. COOP-UC            -> MILP, cooperative
  2. INDI-UC            -> MILP, individual
  3. Greedy-NN          -> greedy_nn, cooperative
  4. Static-Partition   -> greedy_partition_nn, cooperative

Options:
  --num_agents <int>                Number of UAVs to use for every run.
  --num_areas <int>                 Number of areas.
  --num_users <int>                 Number of users per area.
  --env <name>                      Environment type.
  --objective <name>                Objective function.
  --stage_solution <int>            Stage solution to pass through.
  --trials <int>                    Number of trials.
  --enable_ga <yes|no>              Keep existing GA warm-start behavior.
  --solver_backend <name>           MILP backend for MILP runs.
  --solver_time_limit_seconds <n>   Solver time limit.
  --output_dir <path>               Directory for manifests, logs, CSVs, and figures.
  --python <path>                   Python interpreter to use.
  --help                            Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        --num_areas)
            NUM_AREAS="$2"
            shift 2
            ;;
        --num_users)
            NUM_USERS="$2"
            shift 2
            ;;
        --env)
            ENV_TYPE="$2"
            shift 2
            ;;
        --objective)
            OBJECTIVE="$2"
            shift 2
            ;;
        --stage_solution)
            STAGE_SOLUTION="$2"
            shift 2
            ;;
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --enable_ga)
            ENABLE_GA="$2"
            shift 2
            ;;
        --solver_backend)
            SOLVER_BACKEND="$2"
            shift 2
            ;;
        --solver_time_limit_seconds)
            SOLVER_TIME_LIMIT_SECONDS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            print_help >&2
            exit 1
            ;;
    esac
done

RUNS_DIR="assets/results/metrics/runs"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$RUNS_DIR"

MANIFEST_PATH="$OUTPUT_DIR/run_manifest.csv"
SUCCESS_LOG="$OUTPUT_DIR/success.log"
FAILURE_LOG="$OUTPUT_DIR/failures.log"
MASTER_LOG="$OUTPUT_DIR/execution.log"

printf 'label,model_name,scenario,artifact_dir,log_path,status\n' > "$MANIFEST_PATH"
: > "$SUCCESS_LOG"
: > "$FAILURE_LOG"
: > "$MASTER_LOG"

latest_artifact_dir() {
    ls -td "$RUNS_DIR"/*/ 2>/dev/null | head -n 1 | sed 's:/$::'
}

run_case() {
    local label="$1"
    local model_name="$2"
    local scenario="$3"
    local slug
    local artifact_dir
    local case_log
    local status

    slug="$(printf '%s' "$label" | tr '[:upper:]' '[:lower:]' | tr ' /' '__')"
    case_log="$OUTPUT_DIR/${slug}.log"

    {
        echo "=== $label ==="
        echo "model_name=$model_name scenario=$scenario num_agents=$NUM_AGENTS num_areas=$NUM_AREAS num_users=$NUM_USERS env=$ENV_TYPE objective=$OBJECTIVE"
    } | tee -a "$MASTER_LOG"

    if "$PYTHON_BIN" execution_script.py \
        --mode solve \
        --model_name "$model_name" \
        --scenario "$scenario" \
        --objective "$OBJECTIVE" \
        --num_areas "$NUM_AREAS" \
        --num_users "$NUM_USERS" \
        --num_agents "$NUM_AGENTS" \
        --env "$ENV_TYPE" \
        --stage_solution "$STAGE_SOLUTION" \
        --trials "$TRIALS" \
        --enable_ga "$ENABLE_GA" \
        --solver_backend "$SOLVER_BACKEND" \
        --solver_time_limit_seconds "$SOLVER_TIME_LIMIT_SECONDS" \
        2>&1 | tee "$case_log"; then
        status="success"
        artifact_dir="$(latest_artifact_dir)"
        printf '%s\n' "$label -> $artifact_dir" | tee -a "$SUCCESS_LOG" "$MASTER_LOG"
    else
        status="failed"
        artifact_dir=""
        printf '%s\n' "$label" | tee -a "$FAILURE_LOG" "$MASTER_LOG"
    fi

    printf '%s,%s,%s,%s,%s,%s\n' "$label" "$model_name" "$scenario" "$artifact_dir" "$case_log" "$status" >> "$MANIFEST_PATH"
}

run_case "COOP-UC" "milp" "cooperative"
run_case "INDI-UC" "milp" "individual"
run_case "Greedy-NN" "greedy_nn" "cooperative"
run_case "Static-Partition" "greedy_partition_nn" "cooperative"

"$PYTHON_BIN" dummy_app/tools/solver_comparison_report.py \
    --manifest "$MANIFEST_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_agents "$NUM_AGENTS"

echo "Comparison bundle created in $OUTPUT_DIR"
