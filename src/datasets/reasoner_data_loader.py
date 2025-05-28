import sys

from pathlib import Path
import pandas as pd

sys.path.append("src")
from llm.automate_prompts import get_initial_state_mol, get_initial_state_pep, get_initial_state_pro

# def get_state(dataset, prompt, chain_of_thought=True):
#     if dataset == "zinc":
#         return get_initial_state_zinc(
#             prompt,
#             prediction_model=None,
#             reward_model=None,
#             chain_of_thought=chain_of_thought,
#         )
#     else:
#         raise ValueError(f"Unknown dataset {dataset}")
        
def get_state_mol(prompt, prop_name, opt_direction, task_objective, threshold, mol, prop, root_mol, root_prop, root_sim, conversation_type, conversational_LLM, cot, root):
    return get_initial_state_mol(
        prompt=prompt,
        prop_name=prop_name,
        opt_direction=opt_direction,
        task_objective=task_objective,
        threshold=threshold,
        mol=mol,
        prop=prop,
        root_mol=root_mol,
        root_prop=root_prop,
        root_sim=root_sim,
        conversation_type=conversation_type,
        conversational_LLM=conversational_LLM,
        cot=cot,
        root=root,
    )

def get_state_mol(prompt, prop_name, opt_direction, task_objective, threshold, mol, prop, root_mol, root_prop, root_sim, conversation_type, conversational_LLM, cot, root):
    return get_initial_state_mol(
        prompt=prompt,
        prop_name=prop_name,
        opt_direction=opt_direction,
        task_objective=task_objective,
        threshold=threshold,
        mol=mol,
        prop=prop,
        root_mol=root_mol,
        root_prop=root_prop,
        root_sim=root_sim,
        conversation_type=conversation_type,
        conversational_LLM=conversational_LLM,
        cot=cot,
        root=root,
    )
