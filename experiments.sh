#!/bin/bash 

# Possible values for each parameters 
NUM_OF_AREAS=(50 100 150 200)
NUM_OF_AGENTS=(4 6 8)
NUM_OF_USERS=(7 12 15 20) 
ENV_TYPES=("urban")
SCENARIO=("cooperative")
OBJECTIVE=("energy" "coverage")

SUCCESS_LOG="success_runs.log"
FAILURE_LOG="failed_runs.log"
LOG_FILE="mvmtsp_experimentation_log.txt"

# Clear previous logs
echo "Starting MvMTSP Experimentation runs..." > "$LOG_FILE"
> "$SUCCESS_LOG"
> "$FAILURE_LOG"

for scenario in "${SCENARIO[@]}"; do
    for env_type in ${ENV_TYPES[@]}; do
        for objective in ${OBJECTIVE[@]}; do    
            for num_of_areas in ${NUM_OF_AREAS[@]}; do
                for num_of_users in ${NUM_OF_USERS[@]}; do
                    

                    # Construct the command to run the experiment 
                    CMD="python3 execution_script.py --scenario ${scenario} --objective ${objective} --num_areas=${num_of_areas} --num_users=${num_of_users} --env=${env_type}"

                    echo "Executing: $CMD" | tee -a mvmtsp_experimentation_log.txt

                    # Execute th command (waits for completion before starting the next) 
                    eval $CMD 
                    STATUS=$?
                    
                    if [ $STATUS -eq 0 ]; then
                        echo "✅ SUCCESS: $CMD" >> "$SUCCESS_LOG"
                    else
                        echo "❌ FAILED: $CMD (exit code $STATUS)" >> "$FAILURE_LOG"
                        echo "❌ FAILED: $CMD" >> "$LOG_FILE"
                    fi

                    # Log completion 
                    echo "Finished simulation with scenario=$scenario, objective=$objective, num_of_areas=$num_of_areas, num_of_users=$num_of_users, env_type=$env_type" >> "mvmtsp_experimentation_log.txt"
                    echo "------------------------------------------" >> mvmtsp_experimentation_log.txt
                done 
            done 
        done 
    done 
done 

echo "Finished running all experiments!" >> "$LOG_FILE"
