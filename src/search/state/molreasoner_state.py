"""Create a class for a reasoner state"""
import json
import re
import sys
import time

from ast import literal_eval
from copy import deepcopy
from typing import Union, Dict, List

import numpy as np

sys.path.append("src")
from utils.tool import is_valid_smiles, get_prop_function, calculate_tanimoto_similarity, task_specification_dict_peptide
from search.reward.reward_function import get_reward_mol, get_reward_pep, get_reward_pro

class ReasonerState:
    """A class for the search tree state."""
    
    def __init__(
        self,
        template: list,
        # template: List,
        task_id: int,
        root: bool,
        cot: bool,
        reward: float = None,
        drug_type: str = None,
        
        root_mol: str = None,
        # root_prop: dict[str, float] = None,
        root_prop: Dict[str, float] = None,
        
        prop_name: list = None,
        # prop_name: List = None,
        
        # opt_direction: dict[str, str] = None,
        opt_direction: Dict[str, str] = None,
        
        task_objective: str = None,
        # threshold: dict[str, float] = None,
        threshold: Dict[str, float] = None,
        
        conversation_type: str = None,
        conversational_LLM: str = None,
        
        mol: str = None,
        # prop: dict[str, float] = None,
        prop: Dict[str, float] = None,
        
        best_mol: str = None,
        # best_prop: dict[str, float] = None,
        best_prop: Dict[str, float] = None,
        
        prev_mol_list: list[str] = None,
        # prev_mol_list: List[str] = None,
        
        
        prev_prop_list: list[dict[str, float]] = None,
        # prev_prop_list: List[Dict[str, float]] = None,
        
        
        valid_val: int = 1,
        
        messages: list = None,
        # messages: List = None,
        
        suggestion: list[str] = None,
        # suggestion: List[str] = None,
        
        priors_template: str = None,
        root_prompt: str = None,
        info: dict = None,
        # info: Dict = None,
        
    ):
        """Initialize the object"""
        self.template = template
        self.root = root
        self.cot = cot
        self.reward = reward
        self.task_id = task_id
        self.drug_type = drug_type
        if self.drug_type == "peptide":
            if self.task_id < 400:
                _, source_allele_type, _ = task_specification_dict_peptide[self.task_id]
            else:
                _, source_allele_type, _, _ = task_specification_dict_peptide[self.task_id]
            self.source_allele_type = source_allele_type
            
        self.conversation_type = conversation_type
        self.conversational_LLM = conversational_LLM
        
        self.root_mol = root_mol
        self.root_prop = root_prop
        self.prop_name = prop_name
        self.opt_direction = opt_direction
        self.task_objective = task_objective
        self.threshold = threshold
        self.mol = mol
        self.prop = prop

        if self.cot:
            self.reasoning_instruction = "Let's think step by step."
        else:
            if self.conversational_LLM == "llama":
                self.reasoning_instruction = ""
            else:
                self.reasoning_instruction = "No explanation is needed."
                # self.reasoning_instruction = ""

        if best_mol == None:
            self.best_mol = root_mol
        else:
            self.best_mol = best_mol

        if best_prop == None:
            self.best_prop = root_prop
        else:
            self.best_prop = best_prop
        
        if prev_mol_list is None:
            self.prev_mol_list = [mol]
            self.prev_prop_list = [{key: value for key, value in prop.items()}]
        else:
            self.prev_mol_list = prev_mol_list
            self.prev_prop_list = prev_prop_list
        
        self.valid_val = valid_val

        if messages is None:
            self.messages = []
        else:
            self.messages = messages
        
        if suggestion is None:
            self.suggestion = {}
            self.suggestion[self.best_mol] = []
        else:
            self.suggestion = suggestion
        self.priors_template = priors_template

        if self.drug_type == "small_molecule":
            self.parse_answer = parse_molecule
        elif self.drug_type == "peptide":
            self.parse_answer = parse_peptide
        elif self.drug_type == "protein":
            self.parse_answer = parse_protein
        else:
            self.parse_answer = parse_molecule
            
        self.prop_fns = get_prop_function() # self.prop_fns is a list of function
            
        if info is not None:
            self.info = info
        else:
            self.info = {}
            
        if root_prompt is None:
            self.root_prompt = self.get_generation_prompt
        else:
            self.root_prompt = root_prompt

        if self.drug_type == "small_molecule":
            self.reward_fn = get_reward_mol
        elif self.drug_type == "peptide":
            self.reward_fn = get_reward_pep
        elif self.drug_type == "protein":
            self.reward_fn = get_reward_pro
        self.reward = self.reward_fn(self.prop_name, self.root_prop, self.prop, self.valid_val, self.opt_direction, self.threshold)

    @classmethod
    @staticmethod
    def from_dict(incoming_data: dict):
        """create a query state from dictionary"""
        data = deepcopy(incoming_data)
        return ReasonerState(
            template=data.get("template"),
            cot=data.get("cot"),
            root=data.get("root"),
            reward=data.get("reward", None),
            conversation_type=data.get("conversation_type", None),
            conversational_LLM=data.get("conversational_LLM", None),
            
            root_mol=data.get("root_mol", None),
            root_prop=data.get("root_prop", None),
            prop_name=data.get("prop_name", None),
            opt_direction=data.get("opt_direction", None),
            task_objective=data.get("task_objective", None),
            threshold=data.get("threshold", None),
            
            mol=data.get("mol", None),
            prop=data.get("prop", None),
            prev_mol_list=data.get("prev_mol_list", None),
            prev_prop_list=data.get("prev_prop_list", None),
            
            valid_val=data.get("valid_val", None),
            
            messages=data.get("messages", None),
            suggestion=data.get("suggestion", None),
            priors_template=data.get("priors_template", None),
            root_prompt=data.get("root_prompt", None),
            info=deepcopy(data.get("info", {})),
        )

    @property
    def candidates(self):
        """Return the candidate list of the current answer."""
        return(
            [] if self.answer is None else self.parse_answer(self.answer)
        )    
    
    # def priors_prompt(self, num_generate):
        
    #     guidelines = [
    #     "1. Your new state should not contradict those in the current $search_state. ",
    #     "2. Your suggestion should not repeat categories from $search_state. ",
    #     "3. Please avoid using abbreviations and instead provide the full English name of an atom or functional group. ",
    #     "4. Ensure that the other properties of the drug are not significantly altered. ",
    #     "5. Just add one inclusion criteria or one exclusion criteria (pick one action in the action space) to reach new search state. ",
    #     ]
    #     if self.generation_prompt != self.root_prompt:
    #         current_p_a_condition = (
    #             f"$previous_answer = {self.answer}"
    #             f"\n\n$current_prompt = {self.generation_planner_prompt}"
    #             # f"$current_answer = {self.answer}"
    #         )
    #         current_conditions = (
    #             "$root_prompt, $previous_answer, $current_prompt"
    #         )
            
    #         current_prompt_answer = current_p_a_condition
    #         current_conditions = current_conditions
    #     else:
    #         current_conditions = "$root_prompt"
    #         current_prompt_answer = ""
    #         current_conditions = current_conditions
            
    #     guidelines_list = "\n".join([f"{i}) {g}" for i, g in enumerate(guidelines)])
    #     guidelines_string = (
    #         "Your answers should use the following guidelines:\n" + guidelines_list
    #     )
    #     guidelines = guidelines_string
    #     final_task = "Let's think step-by-step, explain your thought process, with scientific justifications, then return your answer as a dictionary mapping from inclusion_criteria, exclusion_criteria to list of suggestion."
    #     prompt = f"""
    #         $root_question = {self.root_prompt}

    #         $search_state: [
    #             "inclusion_criteria": [],
    #             "exclusion_criteria": [],
    #         ].

    #         $action_space: [
    #             "add one new inclusion criteria",
    #             "add one new exclusion criteria",
    #         ]

    #         {current_prompt_answer}
    #         Consider the {current_conditions}. Your task is to suggest possible actions that could achieve the intent of the $root_question.

    #         {guidelines}

    #         {final_task}
    #     """
    #     # print("prompt: ", prompt)
    #     return prompt

    # @property
    def priors_prompt(self, num_generate):
        """Return the priors prompt for the current state"""
        if self.priors_template is None:
            raise ValueError(
                "Cannot generate priors prompt because priors template is None."
            )
        template_entries = {}
        template_entries.update({"root_prompt": self.root_prompt})
        root_property = ""
        threshold = ""
        for i, prop_nm in enumerate(self.prop_name):
            root_property += f"The {prop_nm} of root molecule is: {self.root_prop[prop_nm]}. "
            try:
                threshold += f"You should optimize the {prop_nm} {self.opt_direction[prop_nm]} more than the amount of {self.threshold[prop_nm]}. "
            except:
                threshold += ""
        template_entries.update({"root_property": root_property})
        template_entries.update({"threshold": threshold})
        
        guidelines = [
            
        ]
        print("Is root", self.root)
        print("previous mol list: ", self.prev_mol_list)
        print("previous prop list: ", self.prev_prop_list)

        if self.get_generation_prompt != self.root_prompt:
            previous_property = ""
            prev_prop = self.prev_prop_list[-1]
            for prop_nm, prop_value in prev_prop.items():
                previous_property += f"The {prop_nm} of previous candidate {self.prev_mol_list[-1]} is {prop_value}. "
            current_p_a_condition = (
                f"$previous_messages = {self.messages}"                    
                f"\n\n$previous_property: {previous_property}"
                
                # f"$current_answer = {self.answer}"
            )
            current_conditions = (
                "$root_prompt, $root_property, $threshold, $previous_messages, $previous_property"
            )
            template_entries.update(
                {
                    "previous_prompt_answer": current_p_a_condition,
                    "current_conditions": current_conditions,
                }
            )
        else:
            current_conditions = "$root_prompt, $root_property, $threshold"
            template_entries.update(
                {
                    "previous_prompt_answer": "",
                    "current_conditions": current_conditions,
                }
            )
        guidelines += [
            f"1. You should give a python list named final_suggestion which contains top-{num_generate} suggestion based on the previous information.\n"
            "2. You should learn from the previous experience, especially the substructure change of molecules.\n"
            "3. Your suggestion should not repeat the previous suggestion in $previous prompt.\n"
            # "4. In your suggestion, Please do not show any abbreviation of a atom or functional group. For example, when you need to show 'hydroxyl group', do not show '(OH)' in your suggestion!\n"
            "4. Each of your suggestion should be a sentence of modification instruction rather than a specific molecule.\n",
            "5. Please note that your suggestion should also consider the similarity before and after modification.\n"
        ]
        guidelines_list = "\n".join([f"{i}) {g}" for i, g in enumerate(guidelines)])
        guidelines_string = (
            "Your answers should use the following guidelines:\n" + guidelines_list
        )
        template_entries.update({"guidelines": guidelines_string})
        template_entries.update(
            {
                "final_task": "Take a deep breath and let's think about the goal and guidelines step-by-step\n" 
                "Remember, you should give your reasoning process first and finally return a python list named final_suggestion!"
            }
        )
        prompt = fstr(self.priors_template, template_entries)
        # print("prompt: ", prompt)
        return prompt
        
    
    def process_prior(self, results):
        """Process the results of the prior prompt."""
        if isinstance(results, str):
            prior_answer = results
            usage = None
        else:
            prior_answer = results["answer"]
            usage = results["usage"].get("usage", None)
            
        # action_lists = parse_prior(prior_answer)
        # current_state = {
        #     "inclusion_criteria": [],
        #     "exclusion_criteria": [],
        # }
        # current_state = update_state(current_state, action_lists)
        # action_list = [state_to_suggestion(current_state)]
        action_list = parse_suggestion(prior_answer)
        
        if "priors" not in self.info:
            self.info["priors"] = [
                deepcopy(
                    {
                        "prompt": self.priors_prompt,
                        "answer": prior_answer,
                        "usage": usage,
                        "parsed_actions": action_list,
                    }
                )
            ]
        else:
            self.info["priors"] += [
                deepcopy(
                    {
                        "prompt": self.priors_prompt,
                        "answer": prior_answer,
                        "usage": usage,
                        "parsed_actions": action_list,
                    }
                ),
            ]
        return action_list
    
    def next(self, candidates: list[str]):
    # def next(self, candidates: List[str]):
    
        next_mol = None
        
        for i, can in enumerate(candidates):
            if (can in self.prev_mol_list):
                continue
            next_mol = can
            break

        if next_mol is None:
            return self.prev_mol_list[-1]
        else:
            return next_mol
        
    def process_generation(
        self, results
    ):
        """Process generation answer and store."""
        if isinstance(results, str):
            self.answer = results
            usage = None
        else:
            try:
                self.answer = results["answer"]['content']
            except:
                self.answer = results["answer"]
            usage = results.get("usage", None)
            
        # print(self.answer)
        if self.conversation_type == "multi":
            self.messages.append({"role": "assistant", "content": self.answer})
            
        next_mol = self.next(self.candidates)
        self.mol = next_mol
        if self.drug_type == "small_molecule":
            self.valid_val = is_valid_smiles(self.mol)
        else:
            self.valid_val = 1
        if self.valid_val != 0:
            for prop_nm in self.prop_name:
                if (prop_nm == "tanimoto_similarity" or prop_nm == "levenshtein_similarity"):
                    self.prop[prop_nm] = self.prop_fns[prop_nm](self.root_mol, self.mol)
                else:
                    self.prop[prop_nm] = self.prop_fns[prop_nm](self.mol)
                # define best node under the constraint of reward function
            reward = self.reward_fn(self.prop_name, self.root_prop, self.prop, self.valid_val, self.opt_direction, self.threshold)
            if reward > self.reward:
                self.best_mol = next_mol
                self.best_prop = self.prop
            self.reward = reward    
            
        if self.best_mol not in self.suggestion:
            self.suggestion[self.best_mol] = []
        self.prev_mol_list.append(self.mol)
        self.prev_prop_list.append(self.prop)
        print("mol: ", self.mol)
        print("prop: ", self.prop)
        
        if "generation" not in self.info.keys():
            self.info["generation"] = [
                deepcopy(
                    {
                        "prompt": self.get_generation_prompt, ### bug
                        "system_prompt": self.generation_system_prompt,
                        "answer": self.answer,
                        "candidates_list": self.candidates,
                        "usage": usage,
                    }
                )
            ]
        else:
            self.info["generation"] += [
                deepcopy(
                    {
                        "prompt": self.get_generation_prompt,
                        "system_prompt": self.generation_system_prompt,
                        "answer": self.answer,
                        "candidates_list": self.candidates,
                        "usage": usage,
                    }
                )
            ]
        # print(self.candidates)
        
    def return_next(self) -> "ReasonerState":
        """Return the successor state of self."""
        if self.conversation_type == "multi":
            new_messages = deepcopy(self.messages)
        elif self.conversation_type == "single":
            new_messages = []
        return ReasonerState(
            task_id=self.task_id,
            template=self.template,
            root=False,
            cot=self.cot,
            conversation_type=self.conversation_type,
            conversational_LLM=self.conversational_LLM,
            drug_type=self.drug_type,
            # reward is None
            
            best_mol=self.best_mol,
            best_prop=self.best_prop,
            root_mol=self.root_mol,
            root_prop=self.root_prop,
            prop_name=self.prop_name,
            opt_direction=self.opt_direction,
            task_objective=self.task_objective,
            prop=deepcopy(self.prop),
            threshold=deepcopy(self.threshold),
            prev_mol_list=deepcopy(self.prev_mol_list),
            prev_prop_list=deepcopy(self.prev_prop_list),
            suggestion=deepcopy(self.suggestion),
            messages=new_messages,
            
            priors_template=self.priors_template,
            root_prompt=self.root_prompt,
        )
    
    @property
    def generation_system_prompt(self):
        """Return the system prompt for the generation prompt."""
        return (
            "You are a helpful chemistry expert with extensive knowledge of drug design. "
        )
    
    @property
    def get_generation_prompt(self):
        prompt = ""
        if len(self.suggestion[self.best_mol]) != 0:
            suggestion = self.suggestion[self.best_mol][-1]
        else:
            suggestion = ""
        if suggestion != "":
            suggestion = "You are suggested to do the modification according to the suggestion: " + suggestion
        if self.conversation_type == "single":
            threshold_specific_prompt = ""
            for prop_nm in self.prop_name:
                if "similarity" in prop_nm:
                    continue
                try:
                    threshold_specific_prompt += f"{self.opt_direction[prop_nm]} {prop_nm} by at least {self.threshold[prop_nm]}. "
                except:
                    threshold_specific_prompt += ""
            if self.drug_type == "peptide":
                vals = {
                    "root_mol": self.root_mol,
                    "task_objective": self.task_objective,
                    "source_allele_type": self.source_allele_type,
                    "threshold_specific_prompt": threshold_specific_prompt,
                    "suggestion": suggestion,
                    "reasoning_instruction": self.reasoning_instruction,
                }
            else:
                vals = {
                    "root_mol": self.root_mol,
                    "task_objective": self.task_objective,
                    "threshold_specific_prompt": threshold_specific_prompt,
                    "suggestion": suggestion,
                    "reasoning_instruction": self.reasoning_instruction,
                }
            prompt = fstr(self.template[0], vals)

        elif self.conversation_type == "multi":
            if self.messages == []:
                threshold_specific_prompt = ""
                for prop_nm in self.prop_name:
                    if "similarity" in prop_nm:
                        continue
                    try:
                        threshold_specific_prompt += f"{self.opt_direction[prop_nm]} {prop_nm} by at least {self.threshold[prop_nm]}. "
                    except:
                        threshold_specific_prompt += ""
                if self.drug_type == "peptide":
                    vals = {
                        "root_mol": self.root_mol,
                        "task_objective": self.task_objective,
                        "threshold_specific_prompt": threshold_specific_prompt,
                        "source_allele_type": self.source_allele_type,
                        "suggestion": suggestion,
                        "reasoning_instruction": self.reasoning_instruction,
                    }
                else:
                    vals = {
                        "root_mol": self.root_mol,
                        "task_objective": self.task_objective,
                        "threshold_specific_prompt": threshold_specific_prompt,
                        "suggestion": suggestion,
                        "reasoning_instruction": self.reasoning_instruction,
                    }
                prompt = fstr(self.template[0], vals)
            else:
                vals = {
                    "prev_wrong_mol": self.prev_mol_list[-1],
                    "suggestion": suggestion,
                }
                prompt = fstr(self.template[1], vals)
        return prompt
    
    @property
    def generation_prompt(self):
        """Return the prompt for this state."""
        prompt = self.get_generation_prompt
        if self.messages == []:
            self.messages.append({"role": "system", "content": self.generation_system_prompt})
        self.messages.append({"role": "user", "content": prompt})
        
        return self.messages    

def generate_expert_prompt(
    template, suggestion, valid_smiles_prompt, invalid_smiles_prompt
):
        
    """Generate prompt based on drug edit expert"""
    vals = {
        "suggestion": suggestion,
    }
    return fstr(template, vals)


# def parse_answer(answer: str, num_answers=None):
#     """parse an answer to a list of molecules"""
#     try:
#         final_answer_location = answer.lower().find("final_answer")
#         if final_answer_location == -1:
#             final_answer_location = answer.lower().find("final answer")
#         if final_answer_location == -1:
#             final_answer_location = answer.lower().find("final")
#         if final_answer_location == -1:
#             final_answer_location = 0
            
#         list_start = answer.find("[", final_answer_location)
#         list_end = answer.find("]", list_start)
#         substring = answer[list_start+1:]
#         if '[' in substring:
#             num = substring.count('[')
#             list_start_ = list_start
#             for _ in range(num):
#                 list_start_ = answer.find("[", list_start_+1)
#                 list_end = answer.find("]", list_end+1)
#                 substring = answer[list_start_+1:]
#         try:
#             answer_list = literal_eval(answer[list_start : list_end + 1])
#         except Exception:
#             answer_list = answer[list_start + 1 : list_end]
#             answer_list = [ans.replace("'", "") for ans in answer_list.split(",")]
#         return [ans.replace('"', "").replace("'", "").strip() for ans in answer_list]
#     except:
#         return []
    
def parse_molecule(response):
    pattern = re.compile(r'[0-9BCOHNSOPrIFlanocs@+\.\-\[\]\(\)\\\/%=#$]{10,}')
    output_sequence_list = pattern.findall(response)
    return output_sequence_list

def parse_peptide(response):
    pattern = re.compile('[A-Z]{5,}')
    output_peptide_list = pattern.findall(response)

    new_output_peptide_list = []
    for output_peptide in output_peptide_list:
        if len(output_peptide) < 16 and "X" not in output_peptide:
            new_output_peptide_list.append(output_peptide)
    output_peptide_list = new_output_peptide_list
    return output_peptide_list

def parse_protein(response):
    pattern = re.compile('[A-Z]{5,}')
    output_protein_list = pattern.findall(response)
    new_output_protein_list = []
    for output_protein in output_protein_list:
        new_output_protein_list.append(output_protein)
    output_protein_list = new_output_protein_list
    return output_protein_list


def parse_suggestion(suggestion: str, num_answers=None):
    """parse an answer to a list of suggestion"""
    try:
        final_suggestion_location = suggestion.lower().find("final_suggestion")
        if final_suggestion_location == -1:
            final_suggestion_location = suggestion.lower().find("final suggestion")
        if final_suggestion_location == -1:
            final_suggestion_location = suggestion.lower().find("final")
        if final_suggestion_location == -1:
            final_suggestion_location = 0
        list_start = suggestion.find("[", final_suggestion_location)
        list_end = suggestion.rfind("]", list_start)
        try:
            answer_list = literal_eval(suggestion[list_start : list_end + 1])
            print("answer_list: ", answer_list)
        except Exception:
            answer_list = suggestion[list_start + 1 : list_end]
            answer_list = [ans.replace("'", "") for ans in answer_list.split(",")]
        if all(isinstance(item, str) for item in answer_list):
            return [ans.replace('"', "").replace("'", "").strip() for ans in answer_list]
        elif all(isinstance(item, list) for item in answer_list):
            return [ans[0].replace('"', "").replace("'", "").strip() for ans in answer_list]
    except:
        return []
    
def parse_prior(prior_answer):
    action_lists = {}
    for line in prior_answer.split("{")[-1].split("\n"):
        if ":" in line:
            action, possible_actions = line.split(":")
            action_list = list(
                {
                    s.strip().replace("'", "").replace('"', "").strip()
                    for s in possible_actions.strip()
                    .replace("[", "")
                    .replace("]", "")
                    .split(",")
                    if s.strip().replace("'", "").replace('"', "").strip() != ""
                }
            )  # Use a set for unique elements only
            action_lists[action.strip().strip('"')] = action_list
    
    return action_lists

def update_state(current_state, new_state):
    for key, value in new_state.items():
        if key in current_state:
            if isinstance(new_state[key], list):
                current_state[key].extend(value)
    return current_state

def fstr(fstring_text, vals):
    """Evaluate the provided fstring_text."""
    ret_val = eval(f"""f'''{fstring_text}'''""", vals)
    return ret_val

def convert_to_string(obj: object, indent=1):
    """Convert the given dictionary to a string for prompts."""
    if isinstance(obj, dict):
        new_dict = obj.copy()
        for k, v in obj.items():
            new_dict[k] = convert_to_string(v, indent=indent + 1)
        return json.dumps(new_dict, indent=indent)
    elif isinstance(obj, list):
        new_list = obj.copy()
        for i, v in enumerate(new_list):
            new_list[i] = convert_to_string(v, indent=indent + 1)
        return json.dumps(new_list, indent=indent)
    else:
        return str(obj)

def state_to_suggestion(state):
    suggestion = ""
    for key, value in state.items():
        inclusion_sentence = "The resulting molecule should include "
        exclusion_sentence = "The resulting molecule should exclude "
        if key == 'inclusion_criteria':
            if len(value) != 0:
                inclusion_sentence += ', '.join(value)
                inclusion_sentence += '. '
                suggestion += inclusion_sentence
        if key == 'exclusion_criteria':
            if len(value) != 0:
                exclusion_sentence += ', '.join(value)
                inclusion_sentence += '. '
                suggestion += exclusion_sentence
    return suggestion