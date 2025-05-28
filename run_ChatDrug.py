import json
import pickle
import argparse
import sys
import os
from pathlib import Path
from ChatDrug.task_and_evaluation.Conversational_LLMs_utils import complete
from chatdrug_utils import (
    construct_PDDS_prompt, load_retrieval_DB, retrieve_and_feedback, load_threshold, conversation, conversation_single, sim_molecule, cal_logP, is_valid_smiles, fill_none_with_previous
)
from ChatDrug.task_and_evaluation import task_to_drug, get_task_specification_dict, evaluate, parse

sys.path.append("src")
from search.reward.reward_function import get_reward_mol
from utils.tool import fstr, is_valid_smiles, get_prop_function, calculate_tanimoto_similarity, parse_answer, get_llm_function, examine_complete, get_track, get_task_info

def main(args):
    drug_type, prop_name, opt_direction, task_objective, threshold = get_task_info(constraint=args.constraint, task_id=args.task_id)
    save_dir = Path(args.output_dir)
    save_dir_ = save_dir.joinpath(str(args.task_id))
    version = 1
    log_path = os.path.join(args.log_dir,  f"{args.conversational_LLM}_chatdrug{args.task_id}_{args.constraint}_C_{args.C}_seed_{args.seed}_{args.conversation_type}_v{version}.log")
    save_dir = save_dir_.joinpath(f"{args.conversational_LLM}_chatdrug_{args.constraint}_C_{args.C}_seed_{args.seed}_{args.conversation_type}_v{version}")
    while True:
        if os.path.exists(log_path) and os.path.exists(save_dir):
        # if os.path.exists(save_dir):
            version+=1
            log_path = os.path.join(args.log_dir, f"{args.conversational_LLM}_chatdrug{args.task_id}_{args.constraint}_C_{args.C}_seed_{args.seed}_{args.conversation_type}_v{version}.log")
            save_dir = save_dir_.joinpath(f"{args.conversational_LLM}_chatdrug_{args.constraint}_C_{args.C}_seed_{args.seed}_{args.conversation_type}_v{version}")
        else:
            break
    save_dir.mkdir(parents=True, exist_ok=True)
    f = open(log_path, 'w', encoding="utf-8")
    record = {}
    
    prop_fns = get_prop_function()
    drug_type = task_to_drug(args.task_id)
    task_specification_dict = get_task_specification_dict(args.task_id, args.conversational_LLM)
    input_drug_list, retrieval_DB = load_retrieval_DB(args.val_mol_list, args.task_id, args.seed)
    threshold_dict = load_threshold(drug_type)
    sim_DB_dict = None
    test_example_dict = None
    
    model = None
    tokenizer = None
    
    num_correct = 0
    sim_hit = 0
    sim_amount = 0
    num_all = 0
    
    record_dict = {}
    
    for index, input_drug in enumerate(input_drug_list):
        
        prop_track = [None] * (args.C+2)
        sim_track = [None] * (args.C+2)
        reward_track = [None] * (args.C+2)
        
        print(f">>Sample {index}", file=f)
        
        record[input_drug]={}
        record[input_drug]['skip_conversation_round'] = -1
        record[input_drug]['retrieval_conversation'] = [{'result':i} for i in range((args.C+1))]
        
        messages = [{"role": "system", "content": "You are a helpful chemistry expert with extensive knowledge of drug design."}]
        
        PDDS_prompt = construct_PDDS_prompt(task_specification_dict, input_drug, drug_type, args.task_id, opt_direction, prop_name, threshold)
        messages.append({"role": "user", "content": PDDS_prompt})
        
        """
        if answer == -1, generated_drug_list = None,
        if answer == 0, generated_drug_list = [],
        """

        prop_track[0] = {}
        for prop_nm in prop_name:
            if (prop_nm == "tanimoto_similarity" or prop_nm == "levenshtein_similarity"):
                prop_track[0][prop_nm] = prop_fns[prop_nm](input_drug, input_drug)
            else:
                prop_track[0][prop_nm] = prop_fns[prop_nm](input_drug)
        # prop_track[0] = {prop_nm: prop_fns[prop_nm](input_drug) for prop_nm in prop_name}
        sim_track[0] = 1.0
        reward_track[0] = get_reward_mol(prop_name=prop_name, root_prop=prop_track[0], new_prop=prop_track[0], valid_val=1, opt_direction=opt_direction, threshold=threshold)
        
        print(f"Input Drug value: {prop_track[0]}", file=f)
        for round_index in range((args.C+1)):
            if args.conversation_type == 'single':
                answer, output_drug = conversation_single(
                    messages=messages, model=model, tokenizer=tokenizer, conversational_LLM=args.conversational_LLM,
                    C=args.C, round_index=round_index, trial_index=args.trial_index, task=args.task_id,
                    drug_type=drug_type, input_drug=input_drug, retrieval_DB=retrieval_DB, record=record,
                    logfile=f, fast_protein=args.fast_protein, constraint=args.constraint, 
                    threshold_dict=threshold_dict, sim_DB_dict=sim_DB_dict, test_example_dict=test_example_dict)
                if output_drug is not None:
                    if is_valid_smiles(output_drug):
                        prop_track[round_index+1] = {}
                        for prop_nm in prop_name:
                            if (prop_nm == "tanimoto_similarity" or prop_nm == "levenshtein_similarity"):
                                prop_track[round_index+1][prop_nm] = prop_fns[prop_nm](input_drug, output_drug)
                            else:
                                prop_track[round_index+1][prop_nm] = prop_fns[prop_nm](output_drug)
                        # prop_track[round_index+1] = {prop_nm: prop_fns[prop_nm](output_drug) for prop_nm in prop_name}
                        print(f"Generated Drug value: {prop_track[round_index+1]}", file=f)
                        sim_track[round_index+1] = sim_molecule(input_drug, output_drug)
                        reward_track[round_index+1] = get_reward_mol(prop_name=prop_name, root_prop=prop_track[0], new_prop=prop_track[round_index+1],  valid_val=1, opt_direction=opt_direction, threshold=threshold)
                if answer != 0 or output_drug == None:
                    break
            elif args.conversation_type == 'multi':
                answer, output_drug = conversation(
                    messages=messages, model=model, tokenizer=tokenizer, conversational_LLM=args.conversational_LLM,
                    C=args.C, round_index=round_index, trial_index=args.trial_index, task=args.task_id,
                    drug_type=drug_type, input_drug=input_drug, retrieval_DB=retrieval_DB, record=record,
                    logfile=f, fast_protein=args.fast_protein, constraint=args.constraint, 
                    threshold_dict=threshold_dict, sim_DB_dict=sim_DB_dict, test_example_dict=test_example_dict)
                if output_drug is not None:
                    if is_valid_smiles(output_drug):
                        prop_track[round_index+1] = {}
                        for prop_nm in prop_name:
                            if (prop_nm == "tanimoto_similarity" or prop_nm == "levenshtein_similarity"):
                                prop_track[round_index+1][prop_nm] = prop_fns[prop_nm](input_drug, output_drug)
                            else:
                                prop_track[round_index+1][prop_nm] = prop_fns[prop_nm](output_drug)
                        # prop_track[round_index+1] = {prop_nm: prop_fns[prop_nm](output_drug) for prop_nm in prop_name}
                        print(f"Generated Drug value: {prop_track[round_index+1]}", file=f)
                        sim_track[round_index+1] = sim_molecule(input_drug, output_drug)
                        reward_track[round_index+1] = get_reward_mol(prop_name=prop_name, root_prop=prop_track[0], new_prop=prop_track[round_index+1], valid_val=1, opt_direction=opt_direction, threshold=threshold)
                if answer != 0 or output_drug == None:
                    break
        
        if answer == -1: #answer different from previous but is not valid
            fill_none_with_previous(prop_track)
            fill_none_with_previous(sim_track)
            fill_none_with_previous(reward_track)
            record_dict[input_drug] = [prop_track, sim_track]
            
            f_name = save_dir / 'search_tree_total_info.pkl'
            with open(f_name, 'wb') as file:
                pickle.dump(record_dict, file)
            continue
        elif answer: #answer different from previous and valid
            fill_none_with_previous(prop_track)
            fill_none_with_previous(sim_track)
            fill_none_with_previous(reward_track)
            num_correct += 1
            sim_hit += sim_track[-1]
            best_reward_index = reward_track.index(max(reward_track))
            best_prop = prop_track[best_reward_index]
            sim_amount = get_reward_mol(prop_name=prop_name, root_prop=prop_track[0], new_prop=best_prop,  valid_val=1, opt_direction=opt_direction, threshold=threshold)
            # sim_amount += abs(best_prop - prop_track[0]) * sim_track[-1]
            num_all += 1
        else: #answer is the same as previous
            fill_none_with_previous(prop_track)
            fill_none_with_previous(sim_track)
            fill_none_with_previous(reward_track)
            num_all += 1
            
        print(f'Acc = {num_correct}/{num_all}', file=f)
        print(f'sim_hit = {sim_hit}/{num_all}', file=f)
        print(f'HV = {sim_amount}*{num_correct}/{num_all}', file=f)
        print("----------------", file=f)
    
        
        record_dict[input_drug] = [prop_track, sim_track]

        f_name = save_dir / 'search_tree_total_info.pkl'
        with open(f_name, 'wb') as file:
            pickle.dump(record_dict, file)
        
        
    print("--------Final Acc--------", file=f)
    print(f'Acc = {num_correct}/{num_all}', file=f)
    print("----------------", file=f)

    f_name = save_dir / 'search_tree_total_info.pkl'
    with open(f_name, 'wb') as file:
        pickle.dump(record_dict, file)
    
    record_file = save_dir / "ChatDrug.json"
    with open(record_file, 'w') as rf:
        json.dump(record, rf, ensure_ascii=False)
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--task_id', action='store', required=False, default=102, type=int, help='task_id')
    parser.add_argument('--conversation_type', action='store', required=False, default='multi', type=str, help='conversation_type')
    parser.add_argument('--conversational_LLM', action='store', required=False, type=str, default='deepseek', help='chatgpt, llama, deepseek, dfm')
    parser.add_argument("--val_mol_list", type=str, default='Data/small_molecule_editing_new.txt', help="Path to the validation set")
    parser.add_argument('--log_dir', type=str, default='log', help='directory for saving')
    parser.add_argument('--log_file', action='store', required=False, type=str, default='ChatDrug.log', help='saved log file name')
    parser.add_argument('--output_dir', type=str, default='results/ChatDrug', help='directory for saving')
    parser.add_argument('--constraint', required=False, type=str, default='loose', help='loose or strict')
    parser.add_argument('--seed', required=False, type=int, default=0, help='seed for retrieval data base')
    parser.add_argument('--trial_index', required=False, type=int, default=0, help='trial index for molecule editing tasks')
    parser.add_argument('--C', required=False, type=int, default=2, help='number of conversation round')
    parser.add_argument('--fast_protein', required=False, type=bool, default=False, help='whether to use fast protein evaluation')
    args = parser.parse_args()
    # args = vars(args)

    main(args)