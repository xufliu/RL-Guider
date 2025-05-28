import os
import sys
import time
import json
import pickle
import numpy as np
from pathlib import Path
import aiofiles
import argparse
import asyncio
import nest_asyncio
nest_asyncio.apply()

sys.path.append("src")
from utils.tool import get_prop_function, get_llm_function, examine_complete, examine_complete_fast_protein, get_fast_protein_dict, get_track, get_task_info, load_dataset
from llm.prompt_template import get_generation_prompt_template
from llm.automate_prompts import get_initial_state
from search.state.molreasoner_state import ReasonerState
from search.policy.base_policy import Base_Policy
from search.policy.llm_planner_policy import LLM_Planner_Policy
from search.policy.rl_planner_policy import RL_Planner_Policy
from search.methods.tree_search import SearchTree, init_search_tree
from search.reward.reward_function import reward_function

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        if callable(obj):
            return str(obj)
        return super(NpEncoder, self).default(obj)

async def async_write_log(log_path: Path, content: str):
    async with aiofiles.open(log_path, 'a', encoding='utf-8') as f:
        await f.write(content + '\n')

def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

"""
task id:
10x, 20x: small_molecule
30x, 40x: peptide
50x: protein
"""
async def tree_search(args, tree):
    async def search(args, tree):
        continue_searching = True
        while len(tree) < args.depth and (continue_searching):
            data, answer = tree.step_return()
            if answer == "invalid":
                continue_searching = False
            
            if answer == "duplicate":
                continue_searching = False
    
            if answer == "valid":
                try:
                    if args.fast_protein:
                        if examine_complete_fast_protein(tree.nodes[-1], fast_protein_dict, args.task_id):
                            answer = "complete"
                            continue_searching = False
                    elif examine_complete(search.nodes[-1], args.task_id, args.constraint):
                        answer = "complete"
                        continue_searching = False
                except:
                    answer == "invalid"
                    continue_searching = False
                # print("Current status: ", answer)
                # print("Last layer node reward: ", reward_function(search.nodes[-1]))
                
        prop_track = get_track(tree, args.depth, reward_function)
        return (prop_track, answer)
            
    completions = await search(
        args=args,
        tree=tree,
    )
    return completions

async def tree_searches(
    args, tree_list
):
    completions = [
        tree_search(args, tree) for tree in tree_list
    ]
    
    answers = await asyncio.gather(*completions)
    return answers

def main(args):
    assert isinstance(args.depth, int) and args.depth > 0 
    start = time.time()
    if args.cot:
        cot_str = "cot_"
    else:
        cot_str = ""
    prompt = get_generation_prompt_template(args.task_id, args.conversation_type, args.conversational_LLM)

    drug_type, prop_name, opt_direction, task_objective, threshold = get_task_info(constraint=args.constraint, task_id=args.task_id)
    
    save_dir = Path('results')
    log_dir = Path('log')
    save_dir = save_dir.joinpath(args.planner)
    log_dir = log_dir.joinpath(args.planner)
    save_dir_ = save_dir.joinpath(f"{args.task_id}")
    log_dir_ = log_dir.joinpath(f"{args.task_id}")

    version = 1
    log_dir = log_dir_.joinpath(f"{args.conversational_LLM}_{args.planner}_{args.constraint}_depth_{args.depth}_gen_{args.num_generate}_keep_{args.num_keep}_mol_{args.num_of_mol}_{args.conversation_type}_{cot_str}v{version}.log")
    save_dir = save_dir_.joinpath(f"{args.conversational_LLM}_{args.planner}_{args.constraint}_depth_{args.depth}_gen_{args.num_generate}_keep_{args.num_keep}_mol_{args.num_of_mol}_{args.conversation_type}_{cot_str}v{version}")

    while True:
        if os.path.exists(log_dir) and os.path.exists(save_dir):
            version+=1
            log_dir = log_dir_.joinpath(f"{args.conversational_LLM}_{args.planner}_{args.constraint}_depth_{args.depth}_gen_{args.num_generate}_keep_{args.num_keep}_mol_{args.num_of_mol}_{args.conversation_type}_{cot_str}v{version}.log")
            save_dir = save_dir_.joinpath(f"{args.conversational_LLM}_{args.planner}_{args.constraint}_depth_{args.depth}_gen_{args.num_generate}_keep_{args.num_keep}_mol_{args.num_of_mol}_{args.conversation_type}_{cot_str}v{version}")
        else:
            log_dir_.mkdir(parents=True, exist_ok=True)
            break

    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("Result json path: ", str(save_dir))
    f = open(log_dir, 'w', encoding="utf-8")
    print("args: ", args, file=f)
    
    conversational_LLM_function = get_llm_function(args.conversational_LLM)
    planning_LLM_function = get_llm_function(args.planning_LLM)
    # reward_fn = get_reward_function(args.task_id, args.constraint)
    prop_fns = get_prop_function() # prop_fns is a list of property function
    
    num_of_mol = args.num_of_mol

    num_valid = 0
    num_correct = 0
    num_all = 0

    input_drug_list = load_dataset(args.task_id)
    
    input_drug_list = input_drug_list[:num_of_mol]
    record_dict = {}

    if args.planner == 'baseline':
        policy = Base_Policy(llm_function=planning_LLM_function, log_path=log_dir, log_file=f)
    elif args.planner == 'llm_planner':
        policy = LLM_Planner_Policy(llm_function=planning_LLM_function, log_path=log_dir, log_file=f)
    elif args.planner == 'rl_planner':
        policy = RL_Planner_Policy(drug_type=drug_type, llm_function=planning_LLM_function, log_path=log_dir, rl_model_path=args.rl_model_path, task_id=args.task_id, log_file=f)

    if args.fast_protein:
        assert args.task_id > 500
        fast_protein_dict = get_fast_protein_dict(task_id=args.task_id, input_drug_list=input_drug_list, saved_file='/root/ChatDrug/data/saved_fast_protein_dict')
    else:
        fast_protein_dict = None

    batch_size=args.batch_size
    total_start_time = time.time()
    def run_tree_search(mol_ids):
        mol_list = [input_drug_list[mol_id] for mol_id in mol_ids]
        prop_list = []
        for mol in mol_list:
            prop = {}
            for prop_nm in prop_name:
                if (prop_nm == "tanimoto_similarity" or prop_nm == "levenshtein_similarity"):
                    prop[prop_nm] = prop_fns[prop_nm](mol, mol)
                else:
                    prop[prop_nm] = prop_fns[prop_nm](mol)
            prop_list.append(prop)
        root_mol_list = mol_list
        root_prop_list = prop_list
        tree_list = []
        for idx, mol in enumerate(mol_list):
            starting_state = get_initial_state(task_id=args.task_id, drug_type=drug_type, prompt=prompt, prop_name=prop_name, opt_direction=opt_direction, task_objective=task_objective, threshold=threshold, mol=mol, prop=prop_list[idx], root_mol=root_mol_list[idx], root_prop=root_prop_list[idx], conversation_type=args.conversation_type, conversational_LLM=args.conversational_LLM, cot=args.cot, root=True)
            fname = save_dir / f"search_tree_{mol_ids[idx]}.json"
            if Path(fname).exists() and os.stat(fname).st_size != 0:
                print(f"Loading a tree from {fname}", file=f)
                with open(fname, "r") as file:
                    tree_data = json.load(file)
                    search = SearchTree.from_data(
                        tree_data,
                        conversational_LLM_function,
                        policy,
                        # reward_fn,
                        node_constructor=ReasonerState.from_dict,
                        log_file=f,
                    )
                    assert(
                        isinstance(args.num_keep, int)
                        and args.num_keep == search.num_keep
                    ), "mismatch parameter (num_keep)"
                    assert(
                        isinstance(args.num_generate, int)
                        and args.num_generate == search.num_generate
                    ), "mismatch parameter (num_generate)"
            else:
                search = init_search_tree(args, starting_state, conversational_LLM_function, policy, log_file=f)
            tree_list.append(search)    

        prop_tracks = []
        answers = []
        async def main():
            answer_objects = await tree_searches(args, tree_list)
            for a in answer_objects:
                prop_tracks.append(a[0])
                answers.append(a[1])
        
        asyncio.run(main())
        return prop_tracks, answers

    assert num_of_mol % batch_size == 0
    data = list(range(num_of_mol))
    mol_id_batches = batch_generator(data, batch_size)

    prop_trackss = []
    answerss = []
    tname = save_dir / f"search_tree_total_info.pkl"
    for mol_id_batch in mol_id_batches:
        prop_tracks, answers = run_tree_search(mol_id_batch)
        prop_trackss.extend(prop_tracks)
        answerss.extend(answers)
        for mol_id in mol_id_batch:
            record_dict[input_drug_list[mol_id]] = prop_trackss[mol_id]
            with open(tname, 'wb') as file:
                pickle.dump(record_dict, file)

    for idx, mol in enumerate(input_drug_list):
        if not (answerss[idx] == "invalid"):
            num_valid += 1

        if answerss[idx] == "complete":
            num_correct += 1
            
        if num_all != 0:
            print(f'Acc = {num_correct}/{num_all}', file=f)
            print(f'Hit Ratio = {num_correct}/{num_valid}', file=f)
            print("----------------", file=f)
    
    if num_all != 0:
        print("--------Final Acc--------", file=f)
        print(f'Acc = {num_correct}/{num_all}', file=f)
        print(f'Hit Ratio = {num_correct}/{num_valid}', file=f)
        print("----------------", file=f)

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"Total time used: {total_time}", file=f)
    with open(tname, 'wb') as file:
        pickle.dump(record_dict, file)
                 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Awesome Drug Edit")
    
    # basic param
    # parser.add_argument("--molecule_type", type=str, default='small_molecule', help='small_molecule, peptide, protein')
    # parser.add_argument("--val_mol_list", type=str, default='Data/small_molecule_editing_new.txt', help="Path to the ZINC dataset")
    parser.add_argument("--rl_model_path", type=str, default="results/rl_model_checkpoint/general_replay_buffer_mol_strict_101_best_reward.pth", help="Path to pretrained rl planner model")
    parser.add_argument("--conversational_LLM", type=str, default="deepseek", help="type of LLM (deepseek or chatgpt or llama or dfm)")
    parser.add_argument("--planning_LLM", type=str, default="deepseek", help="type of planning LLM")
    parser.add_argument("--fast_protein", type=bool, default=False, help="whether to use fast protein")
    
    # Experiment setting
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument('--cot', action='store_const', const='cot', help="Run in COT mode")
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
        

