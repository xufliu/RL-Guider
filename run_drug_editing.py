import os
import sys
import time
import json
import pickle
import numpy as np
from pathlib import Path

import argparse

sys.path.append("src")
from utils.tool import get_prop_function, get_llm_function, examine_complete, get_track, get_task_info, NpEncoder
from llm.prompt_template import get_generation_prompt_template
from datasets.reasoner_data_loader import get_state_
from search.state.molreasoner_state import ReasonerState
from search.policy.base_policy import Base_Policy
from search.policy.llm_planner_policy import LLM_Planner_Policy
from search.policy.rl_planner_policy import RL_Planner_Policy
from search.methods.tree_search import SearchTree, init_search_tree
from search.reward.reward_function import get_reward_function

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Awesome Drug Edit")
    
    # basic param
    parser.add_argument("--val_mol_list", type=str, default='Data/small_molecule_editing_new.txt', help="Path to the ZINC dataset")
    parser.add_argument("--rl_model_path", type=str, default="results/rl_model_checkpoint/general_replay_buffer_mol_strict_101_best_reward.pth", help="Path to pretrained rl planner model")
    parser.add_argument("--conversational_LLM", type=str, default="deepseek", help="type of LLM (deepseek or chatgpt or llama)")
    parser.add_argument("--planning_LLM", type=str, default="deepseek", help="type of planning LLM")
    
    # Experiment setting
    parser.add_argument('--cot', action='store_const', const='cot', help="Run in COT mode")
    parser.add_argument("--exact", action='store_const', const='exact', help="Control the property in a certain degree")
    parser.add_argument("--num_of_mol", type=int, default=200, help="Number of molecules to process")
    parser.add_argument("--task_id", type=int, default=102, help="optimization task")
    parser.add_argument("--constraint", type=str, default='loose', help="loose or strict")
    parser.add_argument("--conversation_type", type=str, default="multi", help="multi or single")
    
    # Planner
    parser.add_argument('--planner', type=str, default='baseline', help="type of planner")
    
    # Tree
    parser.add_argument("--depth", type=int, default=1, help="Depth of tree/chain")
    parser.add_argument("--num_generate", type=int, default=5, help="Number of nodes to expand every time.")
    parser.add_argument("--num_keep", type=int, default=1, help="Number of nodes to keep in every expanding.")
    
    args = parser.parse_args()
    
    main(args)