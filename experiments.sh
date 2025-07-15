#!/bin/bash 

# Possible values for each parameters 
NUM_OF_AREAS=(50 100 150 200)
NUM_OF_AGENTS=(4 6)
NUM_OF_USERS=(15 20) 
ENV_TYPES=("urban" "rural" "forest")
SCENARIO=("cooperative")
OBJECTIVE=("energy" "coverage")
SOLUTION_STAGES=(1) # Tow-Stage solution is available but hinders computational performance

SUCCESS_LOG="success_runs_coop.log"
FAILURE_LOG="failed_runs_coop.log"
LOG_FILE="mvmtsp_experimentation_log_coop.txt"

# Clear previous logs
echo "Starting MvMTSP Experimentation runs..." > "$LOG_FILE"
> "$SUCCESS_LOG"
> "$FAILURE_LOG"

for scenario in "${SCENARIO[@]}"; do
    for env_type in ${ENV_TYPES[@]}; do
        for objective in ${OBJECTIVE[@]}; do    
            for num_of_areas in ${NUM_OF_AREAS[@]}; do
                for num_of_users in ${NUM_OF_USERS[@]}; do
                    for num_of_agents in ${NUM_OF_AGENTS[@]}; do
                        for stage in ${SOLUTION_STAGES[@]}; do

                            # Construct the command to run the experiment 
                            CMD="python3 execution_script.py --scenario ${scenario} --objective ${objective} --num_areas=${num_of_areas} --num_users=${num_of_users} --num_agents=${num_of_agents} --env=${env_type} --stage_solution=${stage}"

                            echo "Executing: $CMD" | tee -a mvmtsp_experimentation_log_coop.txt

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
                            echo "Finished simulation with scenario=$scenario, objective=$objective, num_of_areas=$num_of_areas, num_of_users=$num_of_users, num_agents=$num_of_agents, env_type=$env_type --stage_solution=${stage}" >> "mvmtsp_experimentation_log.txt"
                            echo "------------------------------------------" >> mvmtsp_experimentation_log_coop.txt
                        done 
                    done
                done 
            done 
        done 
    done 
done 

echo "Finished running all experiments!" >> "$LOG_FILE"
