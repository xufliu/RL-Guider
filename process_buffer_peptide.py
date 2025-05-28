import sys
import ast
import torch
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.append("src")
from utils.rl_utils import get_peptide_all_general_action_list
from model.rl_planner import ReplayBuffer
from utils.tool import get_prop_function, get_task_info

def str_2_emb(peptide, tokenizer, model):
    inputs = tokenizer(peptide, return_tensors = 'pt')["input_ids"]
    hidden_states = model(inputs)[0]

    # embedding with max pooling
    embedding_max = torch.max(hidden_states[0], dim=0)[0].unsqueeze(0)
    return embedding_max

def main(args):
    drug_type, prop_name, opt_direction, task_objective, threshold = get_task_info(constraint="loose", task_id=args.task_id)
    # prop_name = ["binding_affinity_HLA-B*44:02", "binding_affinity_HLA-C*03:03", "binding_affinity_HLA-B*40:01", "binding_affinity_HLA-B*08:01", "binding_affinity_HLA-B*40:02", "binding_affinity_HLA-C*15:02", "binding_affinity_HLA-C*14:02", "binding_affinity_HLA-A*11:01", "levenshtein_similarity"]
    replay_buffer_name = args.replay_buffer_name

    replay_buffer_path = "results/replay_buffer/" + replay_buffer_name + ".csv"
    output_path = "results/replay_buffer/" + replay_buffer_name + ".pkl"
    replay_buffer_df = pd.read_csv(replay_buffer_path)

    model_name = "/root/peptide_embedding_model/"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    action_list = get_peptide_all_general_action_list()
    action_dict = dict(zip(action_list, range(len(action_list))))
    replay_buffer = ReplayBuffer(500000)

    count = 0
    for index, row in tqdm(replay_buffer_df.iterrows(), total=replay_buffer_df.shape[0], desc="Processing Replay Buffer"):
        count += 1
        # try:
        mol = row['mol']
        action = action_dict[row['action']]
        next_mol = row['next_mol']
        prop_fns = get_prop_function()

        curr_prop = {}
        next_prop = {}
        for prop_nm in prop_name:
            if ("similarity" in prop_nm):
                curr_prop[prop_nm] = prop_fns[prop_nm](mol, mol)
                next_prop[prop_nm] = prop_fns[prop_nm](mol, next_mol)
            else:
                curr_prop[prop_nm] = prop_fns[prop_nm](mol)
                next_prop[prop_nm] = prop_fns[prop_nm](next_mol)
        # curr_prop = {prop_nm: prop_fns[prop_nm](mol) for prop_nm in prop_name}
        # next_prop = {prop_nm: prop_fns[prop_nm](next_mol) for prop_nm in prop_name}
    
        curr_emb = str_2_emb([mol], tokenizer, model).squeeze().detach()
        next_emb = str_2_emb([next_mol], tokenizer, model).squeeze().detach()

        replay_buffer.add(np.array(curr_emb), np.array(curr_emb), action, [1.0, curr_prop, curr_prop, next_prop, 0, 0], np.array(next_emb), False)
        # except:
        #     print("Error smiles string: ", mol)
        #     continue

        if count % 50 == 0:
            with open(output_path, 'wb') as file:
                pickle.dump(replay_buffer, file)

    with open(output_path, 'wb') as file:
        pickle.dump(replay_buffer, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get Replay Buffer for offline Training")
    parser.add_argument("--task_id", type=int, default=301, help="Task ID")
    parser.add_argument("--replay_buffer_name", type=str, default="general_replay_buffer_mol", help="Name of Replay Buffer csv")
    args = parser.parse_args()
    main(args)