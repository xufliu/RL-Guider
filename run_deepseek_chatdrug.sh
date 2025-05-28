#!/bin/bash

task_id_values=(104 105 106 107 108)
constraint_values=('strict' 'loose')
  
for task_id in "${task_id_values[@]}"; do
    for constraint in "${constraint_values[@]}"; do
        # if [ "$task_id" -eq 101 ] && [ "$constraint" == "strict" ]; then
        # echo "Skipping task_id $task_id with constraint $constraint"
        # continue
        # fi
        python run_ChatDrug.py \
          --conversational_LLM='deepseek' \
          --C="2" \
          --task_id="$task_id" \
          --constraint="$constraint"
    done
done