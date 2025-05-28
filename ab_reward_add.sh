python train_rl.py --task_id=101 --replay_buffer_name='general_replay_buffer_mol' --constraint='strict' --reward_type='add' --a=0 --b=1 --c=1 --tau=0.01

python train_rl.py --task_id=101 --replay_buffer_name='general_replay_buffer_mol' --constraint='strict' --reward_type='add' --a=1 --b=0 --c=1 --tau=0.01

python train_rl.py --task_id=101 --replay_buffer_name='general_replay_buffer_mol' --constraint='strict' --reward_type='add' --a=1 --b=1 --c=0 --tau=0.01
