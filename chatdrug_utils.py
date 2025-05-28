import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from ChatDrug.task_and_evaluation import get_task_specification_dict, evaluate, parse
from ChatDrug.task_and_evaluation.Conversational_LLMs_utils import complete

# def construct_PDDS_prompt(task_specification_dict, input_drug, drug_type, task, opt_direction, prop_name, threshold):
#     if drug_type == 'molecule':
#         task_prompt_template = task_specification_dict[task]
#         prompt = task_prompt_template.replace('SMILES_PLACEHOLDER', input_drug)
#         threshold_specific_prompt = ""
#         for prop_nm in prop_name:
#             threshold_specific_prompt += f"{opt_direction[prop_nm]} {prop_nm} by at least {threshold[prop_nm]} "
#         prompt = prompt + threshold_specific_prompt
#         prompt = prompt + " The output molecule should be similar to the input molecule."
#         prompt = prompt + " Give me five molecules in SMILES only and list them using bullet points."
#     return prompt

def construct_PDDS_prompt(task_specification_dict, input_drug, drug_type, task, opt_direction, prop_name, threshold):
    if drug_type == 'molecule':
        task_prompt_template = task_specification_dict[task]
        prompt = task_prompt_template.replace('SMILES_PLACEHOLDER', input_drug)
        prompt = prompt + "?"
        prompt = prompt + " The output molecule should be similar to the input molecule."
        prompt = prompt + " Give me five molecules in SMILES only and list them using bullet points."
        # prompt = prompt + " No explanation is needed. "
    return prompt

def load_dataset(val_mol_list, drug_type, task):
    if drug_type == 'molecule':
        with open(val_mol_list, 'r') as file:
            lines = file.readlines()

        test_data = [line.strip() for line in lines]
    else:
        raise NotImplementedError
    return test_data

def load_retrieval_DB(val_mol_list, task, seed):
    if task < 300:
        drug_type = 'molecule'
        DBfile = 'Data/250k_rndm_zinc_drugs_clean_3.csv'
        # task_specification_dict = get_task_specification_dict(task)
        input_drug_list = load_dataset(val_mol_list, drug_type, task)
        DB = pd.read_csv(DBfile)
        DB = DB[['smiles']]
        DB = DB.rename(columns={"smiles": "sequence"})
        
        for SEQUENCE_TO_BE_MODIFIED in input_drug_list:
            DB = DB[DB['sequence'].str.find(SEQUENCE_TO_BE_MODIFIED)<0]
        DB = DB.sample(10000, random_state=seed)
    return input_drug_list, DB

def fill_none_with_previous(track):
        previous_value = None
        for i in range(len(track)):
            if track[i] is None:
                if previous_value is not None:
                    track[i] = previous_value
            else:
                previous_value = track[i]

def is_valid_smiles(smiles):
    """
    Check if the SMILES valid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception as e:
        return False

def cal_logP(smiles):
    assert is_valid_smiles(smiles)
    mol = Chem.MolFromSmiles(smiles)
    logP = Descriptors.MolLogP(mol)
    return logP

def sim_molecule(smiles1, smiles2):
    """
    Calculate the tanimoto similarity between two SMILES strings.
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES string(s).")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp1 = generator.GetFingerprint(mol1)
    fp2 = generator.GetFingerprint(mol2)
    
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity

def retrieve_and_feedback(task, DB, input_drug, generated_drug, log_file, constraint, threshold_dict):
    sim_DB = DB.copy()
    sim_list = []
    for index, row in sim_DB.iterrows():
        smiles = row['sequence'].replace('\n', '')
        sim = sim_molecule(smiles, generated_drug)
        sim_list.append(sim)
    sim_DB['sim'] = sim_list
    if task < 300:
        sim_DB = sim_DB.sort_values(by=['sim'], ascending=False)
    
    for index, row in sim_DB.iterrows():
        answer = evaluate(input_drug, row['sequence'].replace('\n', ''), task, constraint, log_file, threshold_dict=threshold_dict)
        if answer:
            return row['sequence'].replace('\n', '')
    raise Exception("Sorry, Cannot fined a good one")
    
def load_threshold(drug_type):
    threshold_dict = None
    return threshold_dict

def ReDF(messages, conversational_LLM, round_index, task, drug_type, input_drug, generated_drug, retrieval_DB, record, logfile, fast_protein, constraint, threshold_dict, sim_DB_dict, test_example_dict):
    print(f'Start Retrieval {round_index+1}', file=logfile)
    try:
        closest_drug = retrieve_and_feedback(task, retrieval_DB, input_drug, generated_drug, logfile, constraint, threshold_dict)
    except:
        error = sys.exc_info()
        if error[0] == Exception:
            print('Cannot find a retrieval result.', file=logfile)
            return 0, None
        else:
            print(error, file=logfile)
            print('Invalid drug. Failed to evaluate. Skipped.', file=logfile)
            record[input_drug]['skip_round'] = round_index
            return -1, None
    print("Retrieval Result:" + closest_drug, file=logfile)
    record[input_drug]['retrieval_conversation'][round_index]['retrieval_drug'] = closest_drug
    if conversational_LLM == 'galactica':
        prompt_ReDF = f'Question: Your provided sequence [START_I_SMILES]{generated_drug}[END_I_SMILES] is not correct. We find a sequence [START_I_SMILES]{closest_drug}[END_I_SMILES] which is correct and similar to the {drug_type} you provided. Can you give me a new {drug_type}?\n'
    else:
        prompt_ReDF = f'Your provided sequence {generated_drug} is not correct. We find a sequence {closest_drug} which is correct and similar to the {drug_type} you provided. Can you give me a new {drug_type}?'
    # prompt_ReDF = f"Your provided sequence {generated_drug} could not achieve goal. The molecule closest to the goal so far is current_best_mol = {closest_drug}. Can you give me new molecules?"
    messages.append({"role": "user", "content": prompt_ReDF})
    return 0, generated_drug
    
def conversation(messages, model, tokenizer, conversational_LLM, C, round_index, trial_index, task, drug_type, input_drug, retrieval_DB, record, logfile, fast_protein, constraint, threshold_dict, sim_DB_dict, test_example_dict):
    generated_text = complete(messages, conversational_LLM, round_index)
    messages.append({"role": "assistant", "content": generated_text})
    
    print("----------------", file=logfile)
    print("User:" + messages[2*round_index+1]["content"], file=logfile)
    print("AI:" + generated_text, file=logfile)
    
    record[input_drug]['retrieval_conversation'][round_index]['user'] = messages[2*round_index +1]["content"]
    record[input_drug]['retrieval_conversation'][round_index]['deepseek'] = generated_text
    
    if round_index >= 1:
        closest_drug = record[input_drug]['retrieval_conversation'][round_index-1]['retrieval_drug']
    else:
        closest_drug = None
    
    generated_drug_list = parse(task, input_drug, generated_text, closest_drug)
    
    if generated_drug_list == None:
        # parse answer error
        record[input_drug]['skip_round'] = round_index
        return -1, None
    elif len(generated_drug_list) == 0:
        # parse an empty list
        record[input_drug]['retrieval_conversation'][round_index]['answer'] = 'False'
        return 0, None
    else:
        generated_drug = generated_drug_list[:min(len(generated_drug_list),5)][trial_index]
        print("Generated Result:"+str(generated_drug), file=logfile)
        record[input_drug]['retrieval_conversation'][round_index]['generated_drug'] = generated_drug
    
    # answer = -1: 1.all the mol repeat; 2.generated mol invalid
    # answer = False: did not surpass threshold; answer = True: pass
    
    answer = evaluate(input_drug, generated_drug, task, constraint, logfile, threshold_dict)
    
    if answer == -1:
        record[input_drug]['skip_round'] = round_index
        return -1, None

    print('Evaluation result is: '+str(answer), file=logfile)
    record[input_drug]['retrieval_conversation'][round_index]['answer']=str(answer)
    if answer:
        return 1, generated_drug
    else:
        if round_index < C:
            answer, generated_drug  = ReDF(messages, conversational_LLM, round_index, task, drug_type, input_drug, generated_drug, 
                retrieval_DB, record, logfile, fast_protein, constraint, threshold_dict, sim_DB_dict, test_example_dict)
        return answer, generated_drug
    
def ReDF_single(messages, conversational_LLM, round_index, task, drug_type, input_drug, generated_drug, retrieval_DB, record, logfile, fast_protein, constraint, threshold_dict, sim_DB_dict, test_example_dict):
    print(f'Start Retrieval {round_index+1}', file=logfile)
    try:
        closest_drug = retrieve_and_feedback(task, retrieval_DB, input_drug, generated_drug, logfile, constraint, threshold_dict)
    except:
        error = sys.exc_info()
        if error[0] == Exception:
            print('Cannot find a retrieval result.', file=logfile)
            return 0, None
        else:
            print(error, file=logfile)
            print('Invalid drug. Failed to evaluate. Skipped.', file=logfile)
            record[input_drug]['skip_round'] = round_index
            return -1, None
    print("Retrieval Result:" + closest_drug, file=logfile)
    task_specification_dict = get_task_specification_dict(task=task, conversational_LLM=conversational_LLM)
    task_prompt_template = task_specification_dict[task]
    prompt = task_prompt_template.replace('SMILES_PLACEHOLDER', input_drug)
    prompt = prompt + "?"
    prompt = prompt + " Your output molecule should be similar to the input molecule, which is the molecule to be edited."
    prompt = prompt + f" We find a sequence {closest_drug} which is correct."
    prompt = prompt + " Your output molecule should not be the same to the correct molecule."
    prompt = prompt + " Give me five molecules in SMILES only and list them using bullet points."
    prompt = prompt + " No explanation is needed. "
    record[input_drug]['retrieval_conversation'][round_index]['retrieval_drug'] = closest_drug
    
    # prompt_ReDF = f"Your provided sequence {generated_drug} could not achieve goal. The molecule closest to the goal so far is current_best_mol = {closest_drug}. Can you give me new molecules?"
    messages.pop()
    messages.append({"role": "user", "content": prompt})
    return 0, generated_drug

def conversation_single(messages, model, tokenizer, conversational_LLM, C, round_index, trial_index, task, drug_type, input_drug, retrieval_DB, record, logfile, fast_protein, constraint, threshold_dict, sim_DB_dict, test_example_dict):
    messages.pop()
    print(f'Start Retrieval {round_index+1}', file=logfile)
    try:
        closest_drug = retrieve_and_feedback(task, retrieval_DB, input_drug, input_drug, logfile, constraint, threshold_dict)
    except:
        error = sys.exc_info()
        if error[0] == Exception:
            print('Cannot find a retrieval result.', file=logfile)
            return 0, None
        else:
            print(error, file=logfile)
            print('Invalid drug. Failed to evaluate. Skipped.', file=logfile)
            record[input_drug]['skip_round'] = round_index
            return -1, None
    print("Retrieval Result:" + closest_drug, file=logfile)
    task_specification_dict = get_task_specification_dict(task=task, conversational_LLM=conversational_LLM)
    
    task_prompt_template = task_specification_dict[task]
    prompt = task_prompt_template.replace('SMILES_PLACEHOLDER', input_drug)
    prompt = prompt + "?"
    if conversational_LLM == 'galactica':
        prompt = prompt + f" We find a sequence [START_I_SMILES]{closest_drug}[END_I_SMILES] which is correct and similar to the molecule you provided. Can you give me a new molecule?\n"
    else:
        # prompt = prompt + " Your output molecule should be similar to the input molecule, which is the molecule to be edited."
        prompt = prompt + f" We find a sequence {closest_drug} which is correct and similar to the input molecule. "
        prompt = prompt + " Your output molecule should not be the same to the correct molecule."
        prompt = prompt + " Give me five molecules in SMILES only and list them using bullet points."
    # prompt = prompt + " No explanation is needed. "
    record[input_drug]['retrieval_conversation'][round_index]['retrieval_drug'] = closest_drug
    messages.append({"role": "user", "content": prompt})

    generated_text = complete(messages, conversational_LLM, round_index)
    
    print("----------------", file=logfile)
    print("User:" + messages[-1]["content"], file=logfile)
    print("AI:" + generated_text, file=logfile)
    
    record[input_drug]['retrieval_conversation'][round_index]['user'] = messages[-1]["content"]
    record[input_drug]['retrieval_conversation'][round_index]['deepseek'] = generated_text
    
    generated_drug_list = parse(task, input_drug, generated_text, closest_drug)
    
    if generated_drug_list == None:
        # parse answer error
        record[input_drug]['skip_round'] = round_index
        return -1, None
    elif len(generated_drug_list) == 0:
        # parse an empty list
        record[input_drug]['retrieval_conversation'][round_index]['answer'] = 'False'
        return 0, None
    else:
        generated_drug = generated_drug_list[:min(len(generated_drug_list),5)][trial_index]
        print("Generated Result:"+str(generated_drug), file=logfile)
        record[input_drug]['retrieval_conversation'][round_index]['generated_drug'] = generated_drug
    
    # answer = -1: 1.all the mol repeat; 2.generated mol invalid
    # answer = False: did not surpass threshold; answer = True: pass
    
    answer = evaluate(input_drug, generated_drug, task, constraint, logfile, threshold_dict)
    
    if answer == -1:
        record[input_drug]['skip_round'] = round_index
        return -1, None

    print('Evaluation result is: '+str(answer), file=logfile)
    record[input_drug]['retrieval_conversation'][round_index]['answer']=str(answer)
    if answer:
        return 1, generated_drug
    else:
        # if round_index < C:
        #     answer, generated_drug  = ReDF_single(messages, conversational_LLM, round_index, task, drug_type, input_drug, generated_drug, 
        #         retrieval_DB, record, logfile, fast_protein, constraint, threshold_dict, sim_DB_dict, test_example_dict)
        return answer, generated_drug