python run_planner_tree.py --conversational_LLM='deepseek' --depth=3 --num_generate=1 --num_keep=1 --num_of_mol=200 --task_id=101 --planner='rl_planner' --constraint='strict' --conversation_type='multi' --rl_model_path='results/rl_model_checkpoint/reward_type_mul_a0.5_b0.0_c0.0_tau0.02general_replay_buffer_mol_strict_101_best_reward_v1.pth'

python run_planner_tree.py --conversational_LLM='deepseek' --depth=3 --num_generate=1 --num_keep=1 --num_of_mol=200 --task_id=101 --planner='rl_planner' --constraint='strict' --conversation_type='multi' --rl_model_path='results/rl_model_checkpoint/reward_type_mul_a2.0_b0.0_c0.0_tau0.02general_replay_buffer_mol_strict_101_best_reward_v1.pth'

