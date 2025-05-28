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
from utils.rl_utils import get_smiles_all_general_action_list
from model.rl_planner import ReplayBuffer
from utils.tool import calculate_tanimoto_similarity
from utils.tool import get_prop_function, get_task_info

def str_2_emb(smiles_list, tokenizer, model):

    inputs = tokenizer(smiles_list, return_tensors="pt")

    outputs = model(**inputs)

    embeddings = outputs.last_hidden_state
    smiles_embedding = embeddings[:, 0, :]
    return smiles_embedding

def main(args):
    drug_type, prop_name, opt_direction, task_objective, threshold = get_task_info(constraint="loose", task_id=args.task_id)
    replay_buffer_name = args.replay_buffer_name

    replay_buffer_path = "results/replay_buffer/" + replay_buffer_name + ".csv"
    output_path = "results/replay_buffer/" + replay_buffer_name + ".pkl"
    replay_buffer_df = pd.read_csv(replay_buffer_path)

    model_name = "/root/smiles_embedding_model/"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    action_list = get_smiles_all_general_action_list()
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
        try:
            for prop_nm in prop_name:
                if "similarity" in prop_nm:
                    curr_prop[prop_nm] = prop_fns[prop_nm](mol, mol)
                    next_prop[prop_nm] = prop_fns[prop_nm](mol, next_mol)
                else:
                    curr_prop[prop_nm] = prop_fns[prop_nm](mol)
                    next_prop[prop_nm] = prop_fns[prop_nm](next_mol)
        except Exception as e:
            print(f"Error calculating properties: {e}")
            print(f"Failed property: {prop_nm}")
            continue

        curr_emb = str_2_emb([mol], tokenizer, model).squeeze().detach()
        next_emb = str_2_emb([next_mol], tokenizer, model).squeeze().detach()

        replay_buffer.add(np.array(curr_emb), np.array(curr_emb), action, [1.0, curr_prop, curr_prop, next_prop, 0, 0], np.array(next_emb), False)
        # except:
        #     print("Error smiles string: ", mol)
        #     continue

        if count % 1000 == 0:
            with open(output_path, 'wb') as file:
                pickle.dump(replay_buffer, file)

    with open(output_path, 'wb') as file:
        pickle.dump(replay_buffer, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get Replay Buffer for offline Training")
    parser.add_argument("--task_id", type=int, default=101, help="Task ID")
    parser.add_argument("--replay_buffer_name", type=str, default="general_replay_buffer_mol", help="Name of Replay Buffer csv")
    args = parser.parse_args()
    main(args)