"""Create a class for a reasoner state"""
import json
import logging
import re
import time

from ast import literal_eval
from copy import deepcopy
from typing import Union

import numpy as np

logging.getLogger().setLevel(logging.INFO)

class ReasonerState:
    """A class for the search tree state."""
    
    def __init__(
        self,
        template: dict,
        reward: float = None,
        molecule_name: str = None,
        num_answers: int=3,
        include_list: list[str] = [],
        exclude_list: list[str] = [],
        relation_to_candidate_list: str = "similar to",
        prev_candidate_list: list[str] = [],
        priors_template: str = None,
        root_prompt: str = None,
        info: dict = None,
    ):
        """Initialize the object"""
        self.template = template
        self.reward = reward
        self.molecule_name = molecule_name
        self.num_answers = num_answers
        self.include_list = include_list.copy()
        self.exclude_list = exclude_list.copy()
        self.relation_to_candidate_list = relation_to_candidate_list
        self.prev_candidate_list = prev_candidate_list.copy()
        self.priors_template = priors_template
        if info is not None:
            self.info = info
        else:
            self.info = {}
            
        if root_prompt is None:
            self.root_prompt = self.generation_prompt
        else:
            self.root_prompt = root_prompt
        
    @property
    def candidates(self):
        """Return the candidate list of the current answer."""
        return(
            [] if self.answer is None else parse_answer(self.answer, self.num_answers)
        )
    
    
    @property
    def priors_prompt(self):
        """Return the priors prompt for the current state"""
        if self.priors_template is None:
            raise ValueError(
                "Cannot generate priors prompt because priors template is None."
            )
        current_state = {
            "inclusion_criteria": self.include_list,
            "exclusion_criteria": self.exclude_list,
            "relationship_to_candidate_list": self.relation_to_candidate_list,
        }
        actions_keys = list(current_state.keys())
        actions_descriptions = [
            "add a new inclusion criteria ",
            "add a new exclusion criteria",
            "change the relationship to the candidate list",
        ]
        template_entries = {
            "current_state": convert_to_string(current_state),
            "actions_keys": convert_to_string(actions_keys, indent=0),
            "action_space": convert_to_string(actions_descriptions),
        }
        template_entries.update({"root_prompt": self.root_prompt})
        guidelines = [
            "Your proposed drug whose smile string may be a category similar to, different from, or be a "
            "subclass of previous candidates",
            "Your new category, inclusion criteria, exclusion criteria, and "
            "relationship should not contradict those in the current $search_state.",
        ]
        if self.generation_prompt != self.root_prompt:
            current_p_a_condition = (
                f"$current_prompt = {self.generation_prompt}"
                f"\n\n$current_answer = {self.answer}"
            )
            current_conditions = (
                "$search_state, $root_prompt, $current_question and $current_answer"
            )
            template_entries.update(
                {
                    "root_prompt": self.root_prompt,
                    "current_prompt_answer": current_p_a_condition,
                    "current_conditions": current_conditions,
                }
            )
            guidelines.append(
                "Your suggestions should use scientific explanations from the answers "
                "and explanations in $current_answer"
            )
        else:
            current_conditions = "$search_state and $root_prompt"
            template_entries.update(
                {
                    "root_prompt": self.root_prompt,
                    "current_prompt_answer": "",
                    "current_conditions": current_conditions,
                }
            )
        guidelines += [
            "Your suggestions should not include toxic molecules",
            "Your suggestions should not repeat categories from $search_state",
        ]
        guidelines_list = "\n".join([f"{i}) {g}" for i, g in enumerate(guidelines)])
        guidelines_string = (
            "Your answers should use the following guidelines:\n" + guidelines_list
        )
        template_entries.update({"guidelines": guidelines_string})
        keys_string = ", ".join(['"' + k + '"' for k in list(current_state.keys())])
        template_entries.update(
            {
                "final_task": "Let's think step-by-step, explain your "
                "thought process, with scientific justifications, then return your "
                "answer as a dictionary mapping from "
                f"[{keys_string}] "
                "to lists of suggestions."
            }
        )
        prompt = fstr(self.priors_template, template_entries)
        print("prompt: ", prompt)
        return prompt
        
    
    def process_prior(self, results):
        """Process the results of the prior prompt."""
        if isinstance(results, str):
            prior_answer = results
            usage = None
        else:
            prior_answer = results["answer"]
            usage = results["usage"].get("usage", None)
            
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
        if "priors" not in self.info:
            self.info["priors"] = [
                deepcopy(
                    {
                        "prompt": self.priors_prompt,
                        "answer": prior_answer,
                        "usage": usage,
                        "parsed_actions": action_lists,
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
                        "parsed_actions": action_lists,
                    }
                ),
            ]
        return action_lists
    
    def process_generation(
        self, results
    ):
        """Process generation answer and store."""
        if isinstance(results, str):
            self.answer = results
            usage = None
        else:
            self.answer = results["answer"]
            usage = results.get("usage", None)
        
        if "generation" not in self.info.keys():
            self.info["generation"] = [
                deepcopy(
                    {
                        "prompt": self.generation_prompt,
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
                        "prompt": self.generation_prompt,
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
        return ReasonerState(
            template=self.template,
            priors_template=self.priors_template,
            prev_candidate_list=self.candidates,
            relation_to_candidate_list=self.relation_to_candidate_list,
            include_list=self.include_list.copy(),
            exclude_list=self.exclude_list.copy(),
            root_prompt=self.root_prompt,
        )
    
    @property
    def generation_system_prompt(self):
        """Return the system prompt for the generation prompt."""
        return (
            "You are a helpful chemistry expert with extensive knowledge of drug design. "
            "You will give recommendations for drug editing, including chemically accurate descriptions "
            "and corresponding SMILES string of the drug that you encounter. "
            "Make specific recommendations for designing new drugs based on the given demand, "
            "including their SMILES string representations. "
            "The generated SMILES strings must strictly conform to the SMILES syntax rules."
        )
    
    @property
    def generation_prompt(self):
        """Return the prompt for this state."""
        return generate_expert_prompt(
            template=self.template,
            num_answers=self.num_answers,
            candidate_list=self.prev_candidate_list,
            relation_to_candidate_list=self.relation_to_candidate_list,
            include_list=self.include_list,
            exclude_list=self.exclude_list,
        )
    

class ReasonerState_:
    """A class for the search tree state."""
    
    def __init__(
        self,
        template: dict,
        molecule_name: str = None,
        include_list: list[str] = [],
        exclude_list: list[str] = [],
        relation_to_candidate_list: str = "similar to",
        # prev_candidate_list: list[str] = [],
        prev_prompt: str = "",
        invalid_smiles_prompt: str = "",
        priors_template: str = None,
        root_prompt: str = None,
        info: dict = None,
    ):
        """Initialize the object"""
        self.template = template
        self.molecule_name = molecule_name
        self.include_list = include_list.copy()
        self.exclude_list = exclude_list.copy()
        self.relation_to_candidate_list = relation_to_candidate_list
        # self.prev_candidate_list = prev_candidate_list.copy()
        self.prev_prompt = prev_prompt
        self.invalid_smiles_prompt = invalid_smiles_prompt
        self.priors_template = priors_template
        if info is not None:
            self.info = info
        else:
            self.info = {}
            
        if root_prompt is None:
            self.root_prompt = self.generation_prompt
        else:
            self.root_prompt = root_prompt
        
    @property
    def candidates(self):
        """Return the candidate list of the current answer."""
        return(
            [] if self.answer is None else parse_answer(self.answer)
        )
    
    @property
    def priors_prompt(self):
        """Return the priors prompt for the current state"""
        if self.priors_template is None:
            raise ValueError(
                "Cannot generate priors prompt because priors template is None."
            )
        current_state = {
            "inclusion_criteria": self.include_list,
            "exclusion_criteria": self.exclude_list,
            "relationship_to_candidate_list": self.relation_to_candidate_list,
        }
        actions_keys = list(current_state.keys())
        actions_descriptions = [
            "add a new inclusion criteria ",
            "add a new exclusion criteria",
            "change the relationship to the candidate list",
        ]
        template_entries = {
            "current_state": convert_to_string(current_state),
            "actions_keys": convert_to_string(actions_keys, indent=0),
            "action_space": convert_to_string(actions_descriptions),
        }
        template_entries.update({"root_prompt": self.root_prompt})
        guidelines = [
            "Your proposed drug whose smile string may be a category similar to, different from, or be a "
            "subclass of previous candidates",
            "Your new category, inclusion criteria, exclusion criteria, and "
            "relationship should not contradict those in the current $search_state.",
        ]
        if self.generation_prompt != self.root_prompt:
            current_p_a_condition = (
                f"$current_prompt = {self.generation_prompt}"
                f"\n\n$current_answer = {self.answer}"
            )
            current_conditions = (
                "$search_state, $root_prompt, $current_question and $current_answer"
            )
            template_entries.update(
                {
                    "root_prompt": self.root_prompt,
                    "current_prompt_answer": current_p_a_condition,
                    "current_conditions": current_conditions,
                }
            )
            guidelines.append(
                "Your suggestions should use scientific explanations from the answers "
                "and explanations in $current_answer"
            )
        else:
            current_conditions = "$search_state and $root_prompt"
            template_entries.update(
                {
                    "root_prompt": self.root_prompt,
                    "current_prompt_answer": "",
                    "current_conditions": current_conditions,
                }
            )
        guidelines += [
            "Your suggestions should not include toxic molecules",
            "Your suggestions should not repeat categories from $search_state",
        ]
        guidelines_list = "\n".join([f"{i}) {g}" for i, g in enumerate(guidelines)])
        guidelines_string = (
            "Your answers should use the following guidelines:\n" + guidelines_list
        )
        template_entries.update({"guidelines": guidelines_string})
        keys_string = ", ".join(['"' + k + '"' for k in list(current_state.keys())])
        template_entries.update(
            {
                "final_task": "Let's think step-by-step, explain your "
                "thought process, with scientific justifications, then return your "
                "answer as a dictionary mapping from "
                f"[{keys_string}] "
                "to lists of suggestions."
            }
        )
        prompt = fstr(self.priors_template, template_entries)
        print("prompt: ", prompt)
        return prompt
        
    
    def process_prior(self, results):
        """Process the results of the prior prompt."""
        if isinstance(results, str):
            prior_answer = results
            usage = None
        else:
            prior_answer = results["answer"]
            usage = results["usage"].get("usage", None)
            
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
        if "priors" not in self.info:
            self.info["priors"] = [
                deepcopy(
                    {
                        "prompt": self.priors_prompt,
                        "answer": prior_answer,
                        "usage": usage,
                        "parsed_actions": action_lists,
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
                        "parsed_actions": action_lists,
                    }
                ),
            ]
        return action_lists
    
    def process_generation(
        self, results
    ):
        """Process generation answer and store."""
        if isinstance(results, str):
            self.answer = results
            usage = None
        else:
            self.answer = results["answer"]
            usage = results.get("usage", None)
        
        if "generation" not in self.info.keys():
            self.info["generation"] = [
                deepcopy(
                    {
                        "prompt": self.generation_prompt,
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
                        "prompt": self.generation_prompt,
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
        return ReasonerState_(
            template=self.template,
            priors_template=self.priors_template,
            # prev_candidate_list=self.candidates,
            relation_to_candidate_list=self.relation_to_candidate_list,
            include_list=self.include_list.copy(),
            exclude_list=self.exclude_list.copy(),
            root_prompt=self.root_prompt,
            prev_prompt = self.prev_prompt,
            invalid_smiles_prompt = self.invalid_smiles_prompt
        )
    
    @property
    def generation_system_prompt(self):
        """Return the system prompt for the generation prompt."""
        return (
            "You are a helpful chemistry expert with extensive knowledge of drug design. "
            "You will give recommendations for drug editing, including chemically accurate descriptions "
            "and corresponding SMILES string of the drug that you encounter. "
            "Make specific recommendations for designing new drugs based on the given demand, "
            "including their SMILES string representations. "
            "The generated SMILES strings must strictly conform to the SMILES syntax rules."
        )
    
    @property
    def generation_prompt(self):
        """Return the prompt for this state."""
        return generate_expert_prompt_(
            template=self.template,
            # candidate_list=self.prev_candidate_list,
            relation_to_candidate_list=self.relation_to_candidate_list,
            include_list=self.include_list,
            exclude_list=self.exclude_list,
            prev_prompt=self.prev_prompt,
            invalid_smiles_prompt=self.invalid_smiles_prompt
        )
     
def generate_expert_prompt(
    template, num_answers, candidate_list, relation_to_candidate_list, include_list, exclude_list
):
        
    """Generate prompt based on drug edit expert"""
    if len(candidate_list) != 0 and relation_to_candidate_list is not None:
        candidate_list_statement = "\n\nYou should start with the following list: "
        candidate_list_statement += (
            "["
            + ", ".join(
                [
                    "'" + cand.replace("'", "").replace('"', "").strip() + "'"
                    for cand in candidate_list
                ]
            )
            + "]. "
        )
        candidate_list_statement += f"The list that you return should probably should not have the same drug as this list! "
        candidate_list_statement += f"Your list of drug may be {relation_to_candidate_list} those in the list. "
        candidate_list_statement += (
            "Please compare your list to some of the candidates in this list."
        )
    elif len(candidate_list) != 0 and relation_to_candidate_list is None:
        raise ValueError(
            f"Non-empty candidate list {candidate_list} given with "
            "relation_to_candidate_list == None"
        )
    else:
        candidate_list_statement = ""
    if len(include_list) != 0:
        include_statement = (
            f"You should include candidate drug "
            "with the following properties: "
        )
        include_statement += ", ".join(include_list)
        include_statement += ". "
    else:
        include_statement = ""
    if len(exclude_list) != 0:
        exclude_statement = (
            f"You should exclude candidate drug "
            "with the following properties: "
        )

        exclude_statement += ", ".join(exclude_list)
        exclude_statement += ". "
    else:
        exclude_statement = ""
    
    vals = {
        "candidate_list_statement": candidate_list_statement,
        "include_statement": include_statement,
        "exclude_statement": exclude_statement,
    }
    return fstr(template, vals)

def generate_expert_prompt_(
    template, relation_to_candidate_list, include_list, exclude_list, prev_prompt, invalid_smiles_prompt
):
        
    """Generate prompt based on drug edit expert"""
    prev_prompt_statement = prev_prompt
    invalid_smiles_prompt_statement = invalid_smiles_prompt
    
    
    if len(include_list) != 0:
        include_statement = (
            f"You should include candidate drug "
            "with the following properties: "
        )
        include_statement += ", ".join(include_list)
        include_statement += ". "
    else:
        include_statement = ""
        
        
    if len(exclude_list) != 0:
        exclude_statement = (
            f"You should exclude candidate drug "
            "with the following properties: "
        )

        exclude_statement += ", ".join(exclude_list)
        exclude_statement += ". "
    else:
        exclude_statement = ""
    
    vals = {
        "prev_prompt": prev_prompt_statement,
        "invalid_smiles_prompt": invalid_smiles_prompt_statement,
        "include_statement": include_statement,
        "exclude_statement": exclude_statement,
    }
    return fstr(template, vals)

def parse_answer(answer: str, num_answers=None):
        """parse an answer to a list of molecules"""
        try:
            final_answer_location = answer.lower().find("final_answer")
            if final_answer_location == -1:
                final_answer_location = answer.lower().find("final answer")
            if final_answer_location == -1:
                final_answer_location = answer.lower().find("final")
            if final_answer_location == -1:
                final_answer_location = 0
            list_start = answer.find("[", final_answer_location)
            list_end = answer.find("]", list_start)
            try:
                answer_list = literal_eval(answer[list_start : list_end + 1])
            except Exception:
                answer_list = answer[list_start + 1 : list_end]
                answer_list = [ans.replace("'", "") for ans in answer_list.split(",")]
            return [ans.replace('"', "").replace("'", "").strip() for ans in answer_list]
        except:
            return []


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
    