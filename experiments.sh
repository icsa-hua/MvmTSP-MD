#!/bin/bash 

# Possible values for each parameters 
NUM_OF_AREAS = ("35", "75", "100", "150")
NUM_OF_AGENTS = ("2", "4", "6", "8")
NUM_OF_USERS = ("7", "12", "15","20") 
ENV_TYPES = ("urban","rural","forest")
SCENARIO = ("cooperative","individual")
OBJECTIVE = ("energy","coverage")

echo "Starting MvMTSP Experimentation runs..." > mvmtsp_experimentation_log.txt 

for scenario in "${SCENARIO[@]}"; do
    for objective in ${OBJECTIVE[@]}; do    
        for num_of_areas in ${NUM_OF_AREAS[@]}; do
            for num_of_users in ${NUM_OF_USERS[@]}; do
              

                # Construct the command to run the experiment 
                CMD="python3 execution_script.py --scenario ${scenario} --objective ${objective} --num_of_areas ${num_of_areas} --num_of_users ${num_of_users}"

                echo "Executing: $CMD" | tee -a mvmtsp_experimentation_log.txt

                # Execute th command (waits for completion before starting the next) 
                eval $CMD 

                # Log completion 
                echo "Finished simulation with scenario=$scenario, objective=$objective, num_of_areas=$num_of_areas, num_of_users=$num_of_users" >> "mvmtsp_experimentation_log.txt"
                echo "------------------------------------------" >> mvmtsp_experimentation_log.txt
            done 
        done 
    done 
done 

echo "Finished running all experiments!" >> mvmtsp_experimentation_log.txt
