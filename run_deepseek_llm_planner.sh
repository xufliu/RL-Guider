#!/bin/bash

task_id_values=(104 105 106 107 108)
planner_values=('llm_planner')
constraint_values=('strict' 'loose')
conversation_type_values=('single' 'multi')
  
for task_id in "${task_id_values[@]}"; do
    for planner in "${planner_values[@]}"; do
        for constraint in "${constraint_values[@]}"; do
            # if [ "$task_id" -eq 101 ] && [ "$constraint" == "strict" ]; then
            #     echo "Skipping task_id $task_id with constraint $constraint"
            #     continue
            # fi
            # if [ "$task_id" -eq 102 ] && [ "$constraint" == "strict" ]; then
            #     echo "Skipping task_id $task_id with constraint $constraint"
            #     continue
            # fi
            for conversation_type in "${conversation_type_values[@]}"; do
                python run_planner_tree.py \
                  --conversational_LLM='deepseek' \
                  --depth="3" \
                  --num_generate="1" \
                  --num_keep="1" \
                  --num_of_mol="200" \
                  --task_id="$task_id" \
                  --planner="$planner" \
                  --constraint="$constraint" \
                  --conversation_type="$conversation_type"
            done
        done
    done
done