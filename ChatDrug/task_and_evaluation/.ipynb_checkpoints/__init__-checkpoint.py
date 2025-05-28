import numpy as np
from .small_molecule_editing import evaluate_molecule, task_specification_dict_molecule, parse_molecule, task2threshold_list

def task_to_drug(task):
    if task < 300:
        return 'molecule'
    elif task < 500:
        return 'peptide'
    elif task < 600:
        return 'protein'
    else:
        raise NotImplementedError

def get_task_specification_dict(task):
    if task < 300:
        return task_specification_dict_molecule

    
def parse(task, input_drug, generated_text, addition_drug=None):
    if task < 300:
        return parse_molecule(input_drug, generated_text, addition_drug)
    else:
        raise NotImplementedError

def evaluate(input_drug, generated_drug, task, constraint, log_file, threshold_dict):
    if task < 300:
        if constraint == 'loose':
            threshold_list = task2threshold_list[task][0]
        else:
            threshold_list = task2threshold_list[task][1]
        _, _, answer = evaluate_molecule(input_drug, generated_drug, task, log_file=log_file, threshold_list=threshold_list)
    return answer
