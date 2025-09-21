from openai import OpenAI
import collections
import numpy as np
import pickle
import random
import os
import sys
from pathlib import Path
import pandas as pd

import argparse

sys.path.append("src")
from utils.tool import fstr, is_valid_smiles, calculate_tanimoto_similarity, parse
from utils.rl_utils import get_general_action_list, get_all_general_action_list
from llm.deepseek_interface import run_deepseek_prompts
from llm.chatgpt_interface  import run_openai_prompts
from llm.prompt_template import system_prompt


def main(args):
    all_action_list = get_all_general_action_list()
    log_path = os.path.join(args.log_dir, f"general_replay_buffer_mol.log")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"general_replay_buffer_mol.csv"
    action_count = {ac: 0 for ac in all_action_list}
    if output_path.exists():
        replay_df = pd.read_csv(output_path)
        df_actions = replay_df['action'].tolist()
        for action in df_actions:
            action_count[action] += 1
    else:
        replay_df = pd.DataFrame(columns=['mol', 'action', 'next_mol'])

    f = open(log_path, 'w')
    num_of_episode = args.num_of_episode
    data_path = args.data_path

    zinc_df = pd.read_csv(data_path)
    zinc_df['smiles'] = zinc_df['smiles'].str.replace('\n', '')
    zinc_df['smiles'] = zinc_df['smiles'].str.replace('\r', '')
    smiles_list = zinc_df['smiles'].to_list()

    generation_prompt = (
        "Edit the molecule {mol} by following the suggestion: {suggestion} "
        "Give me five molecules in SMILES only and list them using bullet points. "
    )

    for action in all_action_list:
        print(f"Now run action: {action}")
        num_mol_for_action = action_count[action]
        if num_mol_for_action < num_of_episode:
            print("Start Process")
            gap = num_of_episode - num_mol_for_action
            gap_mol_list = []
            loop = 0
            while len(gap_mol_list) < gap and loop<100:
                if loop % 10 == 0:
                    print(f"loop {loop}. current gap: {gap - len(gap_mol_list)}")
                loop += 1
                temp_mol_list = random.sample(smiles_list, 100)
                for temp_mol in temp_mol_list:
                    temp_mol_action_list = get_general_action_list(temp_mol)
                    if action in temp_mol_action_list:
                        gap_mol_list.append(temp_mol)
                        if len(gap_mol_list) > gap:
                            break
            if loop >= 100:
                print(f"giving random mol for action: {action}.")
                gap_mol_list.extend(random.sample(smiles_list, num_of_episode - len(gap_mol_list)))
            messages_list = []
            messages_idx = []
            for mol_id, mol in enumerate(gap_mol_list):
                messages = [{"role": "system", "content": system_prompt}]
                vals = {
                    "mol": mol,
                    "suggestion": action,
                }
                prompt = fstr(generation_prompt, vals)
                messages.append({"role": "user", "content": prompt})
                messages_list.append(messages)
                messages_idx.append(mol_id)
            if args.llm == 'deepseek':
                answers = run_deepseek_prompts(messages_list)
            elif args.llm == 'chatgpt':
                answers = run_openai_prompts(messages_list)
            print("OK")
            for i, answer in enumerate(answers):
                mol_id = messages_idx[i]
                answer_txt = answer['answer']
                try:
                    answer_list = parse(answer_txt)
                except:
                    print("Fail to parse answer, retry", file=f)
                    print("-"*100, file=f)
                    continue
                sim = 0
                next_mol = ''

                for answer_id, answer_mol in enumerate(answer_list): 
                    if len(answer_mol) > 200 or (answer_mol == ''):
                        continue

                    if is_valid_smiles(answer_mol):
                        sim = calculate_tanimoto_similarity(gap_mol_list[mol_id], answer_mol)
                        if sim == 1.0:
                            continue

                        next_mol = answer_mol
                        new_data = pd.DataFrame([[gap_mol_list[mol_id], action, next_mol]], columns=['mol', 'action', 'next_mol'])
                        replay_df = pd.concat([replay_df, new_data], ignore_index=True)
                        break
                    else:
                        continue
        print("Process Done and save")
        replay_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get Replay Buffer for offline Training")
    parser.add_argument("--llm", type=str, default="deepseek", help="type of LLM")
    parser.add_argument("--data_path", type=str, default="Data/250k_rndm_zinc_drugs_clean_3.csv", help="Path to the ZINC dataset")
    parser.add_argument("--num_of_episode", type=int, default=50, help="Number of episodes to be collected")
    parser.add_argument("--log_dir", type=str, default="log", help="Path to the log file")
    parser.add_argument("--output_dir", type=str, default="results/replay_buffer", help="Path to the output file")
    args = parser.parse_args()
    main(args)

