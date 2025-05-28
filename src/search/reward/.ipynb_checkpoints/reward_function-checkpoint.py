"""Function for getting state reward"""
import logging
import sys
import numpy as np

sys.path.append("src")
from utils.tool import is_valid_smiles, task2threshold_list


# Type 0: Add (1, 1, 1)
# Type 1: Add (0.5, 1, 1)
# Type 2: Add (1, 0.5, 1)
# Type 3: Add (1, 1, 0.5)
# Type 4: Add (0.5, 0.5, 1)
# Type 5: Add (0.5, 1, 0.5)
# Type 6: Add (1, 0.5, 0.5)
# Type 7: Add (0.5, 0.5, 0.5)
# Type 8: Add (1, 1, 1)
# Type 9: Add (1, 1, 1)


def get_reward_mol_train(prop_name, root_prop, new_prop, valid_val, opt_direction, threshold, reward_type, a=1, b=1, c=1):
    
    if reward_type == "add":
        reward = 0
        reward += b*valid_val
        for prop_nm in prop_name:
            if "similarity" in prop_nm:
                reward += a*new_prop[prop_nm]
                continue
            if prop_nm == "tPSA":
                a = 0.05
            elif prop_nm == "QED":
                a = 5
            elif prop_nm == "HBA":
                a = 0.5
            elif prop_nm == "HBD":
                a = 0.5
            if opt_direction[prop_nm] == 'decrease':
                reward += c*(root_prop[prop_nm] - (new_prop[prop_nm] + threshold[prop_nm]))
            elif opt_direction[prop_nm] == 'increase':
                reward += c*(new_prop[prop_nm] - (root_prop[prop_nm] + threshold[prop_nm]))
    elif reward_type == "mul":
        reward = 1
        if b != 0:
            reward = valid_val*reward
        for prop_nm in prop_name:
            if "similarity" in prop_nm:
                if a != 0:
                    if new_prop[prop_nm] == 1:
                        new_prop[prop_nm] = 0
                    reward = new_prop[prop_nm]*reward
                continue
            if prop_nm == "tPSA":
                a = 0.05
            elif prop_nm == "QED":
                a = 5
            elif prop_nm == "HBA":
                a = 0.5
            elif prop_nm == "HBD":
                a = 0.5
            if opt_direction[prop_nm] == 'decrease':
                if c != 0:
                    reward = reward*(root_prop[prop_nm] - (new_prop[prop_nm] + threshold[prop_nm]))
            elif opt_direction[prop_nm] == 'increase':
                if c != 0:
                    reward = reward*(new_prop[prop_nm] - (root_prop[prop_nm] + threshold[prop_nm]))
    return reward



def get_reward_mol(prop_name, root_prop, new_prop, valid_val, opt_direction, threshold, alpha=1):
    reward = 0
    for prop_nm in prop_name:
        if "similarity" in prop_nm:
            continue
        if prop_nm == "tPSA":
            alpha = 0.05
        elif prop_nm == "QED":
            alpha = 5
        elif prop_nm == "HBA":
            alpha = 0.5
        elif prop_nm == "HBD":
            alpha = 0.5
        if opt_direction[prop_nm] == 'decrease':
            reward += (root_prop[prop_nm] - new_prop[prop_nm])*alpha
            reward += (root_prop[prop_nm] - (new_prop[prop_nm] + threshold[prop_nm]))*alpha
        elif opt_direction[prop_nm] == 'increase':
            reward += (new_prop[prop_nm] - root_prop[prop_nm])*alpha
            reward += (new_prop[prop_nm] - (root_prop[prop_nm] + threshold[prop_nm]))*alpha
    return reward

# def get_reward_mol(prop_name, root_prop, new_prop, valid_val, opt_direction, threshold, alpha=1):
#     reward = 0
#     for prop_nm in prop_name:
#         if "similarity" in prop_nm:
#             continue
#         if prop_nm == "tPSA":
#             alpha = 0.05
#         elif prop_nm == "QED":
#             alpha = 5
#         elif prop_nm == "HBA":
#             alpha = 0.5
#         elif prop_nm == "HBD":
#             alpha = 0.5
#         if opt_direction[prop_nm] == 'decrease':
#             down = (root_prop[prop_nm] - new_prop[prop_nm])>0
#             down_pass = (root_prop[prop_nm] - (new_prop[prop_nm] + threshold[prop_nm]))>0
#             if down:
#                 reward += 1
#             if down_pass:
#                 reward += 1
#             if not (down or down_pass):
#                 reward -= 1
#         elif opt_direction[prop_nm] == 'increase':
#             up = (new_prop[prop_nm] - root_prop[prop_nm])>0
#             up_pass = (new_prop[prop_nm] - (root_prop[prop_nm] + threshold[prop_nm]))>0
#             if up:
#                 reward += 1
#             if up_pass:
#                 reward += 1
#             if not (up or up_pass):
#                 reward -= 1
#     return reward


def get_reward_pep(prop_name, root_prop, new_prop, valid_val, opt_direction, threshold, EPS=1e-10):
    reward = 0
    for prop_nm in prop_name:
        if "similarity" in prop_nm:
            continue
        reward += (new_prop[prop_nm] - (root_prop[prop_nm] + EPS))
        reward += (new_prop[prop_nm] - threshold[prop_nm])
    return reward

def get_reward_pro(prop_name, root_prop, new_prop, valid_val, opt_direction, threshold):
    reward = 0
    for prop_nm in prop_name:
        if "similarity" in prop_nm:
            continue
        reward += (new_prop[prop_nm] - root_prop[prop_nm])
    return reward


# list of state
def reward_function(states):
    rewards = [None] * len(states)
    
    for i, s in enumerate(states):
        mol = s.mol
        drug_type = s.drug_type
        if drug_type == "small_molecule":
            if not is_valid_smiles(mol):
                reward = 0
                s.reward = reward
                rewards[i] = reward
            else:
                valid_val = s.valid_val
                prop_name = s.prop_name
                root_prop = s.root_prop
                threshold = s.threshold
                prop = s.prop
                opt_direction = s.opt_direction
                reward = get_reward_mol(prop_name=prop_name, root_prop=root_prop, new_prop=prop, valid_val=valid_val, opt_direction=opt_direction, threshold=threshold)
                s.reward = reward
                rewards[i] = reward
        elif drug_type == "peptide":
            valid_val = s.valid_val
            prop_name = s.prop_name
            root_prop = s.root_prop
            threshold = s.threshold
            prop = s.prop
            opt_direction = s.opt_direction
            reward = get_reward_pep(prop_name=prop_name, root_prop=root_prop, new_prop=prop, valid_val=valid_val, opt_direction=opt_direction, threshold=threshold)
            s.reward = reward
            rewards[i] = reward
        elif drug_type == "protein":
            valid_val = s.valid_val
            prop_name = s.prop_name
            root_prop = s.root_prop
            prop = s.prop
            threshold = s.threshold
            opt_direction = s.opt_direction
            reward = get_reward_pro(prop_name=prop_name, root_prop=root_prop, new_prop=prop, valid_val=valid_val, opt_direction=opt_direction, threshold=threshold)
            s.reward = reward
            rewards[i] = reward
    return rewards


# def base_reward_function_strict(states, threshold):
#     rewards = [None] * len(states)
    
#     for i, s in enumerate(states):
#         mol = s.mol
#         if not is_valid_smiles(mol):
#             reward = 0
#             s.reward = reward
#             rewards[i] = reward
#         else:
#             valid_val = s.valid_val
#             root_prop = s.root_prop
#             prop = s.prop
#             opt_direction = s.opt_direction
#             root_sim = s.root_sim
#             reward = get_thresh_reward(root_prop=root_prop, new_prop=prop, root_sim=root_sim, valid_val=valid_val, opt_direction=opt_direction, threshold=threshold)
#             s.reward = reward
#             rewards[i] = reward
#     return rewards


def get_reward_function(task_id, constraint):
    if constraint == 'strict':
        threshold_idx = 1
    elif constraint == 'loose':
        threshold_idx = 0
    if task_id < 300:
        threshold = task2threshold_list[task_id][threshold_idx][0]
    
    return lambda states: reward_function(states, threshold=threshold)

        
        
        
