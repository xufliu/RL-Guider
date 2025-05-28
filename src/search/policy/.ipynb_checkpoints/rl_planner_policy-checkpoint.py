"""Class for the RL planner policy"""
import traceback
import re
import sys
import time
import torch
import random
from ast import literal_eval
from transformers import BertModel, BertTokenizer

from collections.abc import Callable

import numpy as np

sys.path.append("src")
from search.state.molreasoner_state import ReasonerState
from model.rl_planner import RL_Planner, PPO
from utils.rl_utils import get_peptide_general_action_list, get_peptide_all_general_action_list, get_protein_general_action_list, get_protein_all_general_action_list, get_smiles_general_action_list, get_smiles_all_general_action_list
from llm.deepseek_interface import run_deepseek_prompts
from llm.prompt_template import system_prompt
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorWithPadding
from transformers import AutoModel, AutoTokenizer
from utils.tool import fstr
from search.policy.utils import (
    BabyPolicy, ActionAdder
)

refine_template = """
    $question = {query}\n
    Here is a plain suggestion for this query: $suggestion = {suggestion}.\n
    Do not change the demand of the suggestion and rephrase this suggestion to make it more suitable for solving $question.\n
    You should give a python list named final_suggestion which contains a sentence of suggestion.\n
    Take a deep breath and let's think step-by-step.\n
    Remember, you should return a python list named final_suggestion!
"""

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
        list_end = suggestion.find("]", list_start)
        try:
            answer_list = literal_eval(suggestion[list_start : list_end + 1])
        except Exception:
            answer_list = suggestion[list_start + 1 : list_end]
            answer_list = [ans.replace("'", "") for ans in answer_list.split(",")]
        return [ans.replace('"', "").replace("'", "").strip() for ans in answer_list]
    except:
        return []
    
def str_2_emb_smiles(smiles_list, tokenizer, model):

    inputs = tokenizer(smiles_list, return_tensors="pt")

    outputs = model(**inputs)

    embeddings = outputs.last_hidden_state
    smiles_embedding = embeddings[:, 0, :]
    return smiles_embedding

def str_2_emb_peptide(peptide, tokenizer, model):
    inputs = tokenizer(peptide, return_tensors = 'pt')["input_ids"]
    hidden_states = model(inputs)[0]

    # embedding with max pooling
    embedding_max = torch.max(hidden_states[0], dim=0)[0].unsqueeze(0)
    return embedding_max

def str_2_emb_protein(protein, tokenizer, model):
    protein = " ".join(list(protein))
    protein = re.sub(r"[UZOB]", "X", protein)
    encoded_protein = tokenizer(protein, return_tensors='pt')
    output = model(**encoded_protein)
    embedding = output.last_hidden_state[:, 0, :]
    return embedding

class RL_Planner_Policy(BabyPolicy):
    """RL as Planner"""
    def __init__(
        self,
        drug_type,
        log_file,
        llm_function: callable = lambda list_x: [example_output] * len(list_x),
        max_attempts: int=3,
        log_path: str="",
        rl_model_path: str="",
        task_id: int=None,
    ):
        self.drug_type = drug_type
        self.log_file = log_file
        self.llm_function = llm_function
        self.max_attempts = max_attempts
        self.log_path = log_path
        self.rl_model_path = rl_model_path
        self.task_id = task_id
        
        self.load_module()
        
    def load_module(self):
        if self.drug_type == "small_molecule":
            model_name = "/root/smiles_embedding_model/"
            self.str_2_emb = str_2_emb_smiles
            self.action_list = get_smiles_all_general_action_list()
            self.drug_action_list = get_smiles_general_action_list
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif self.drug_type == "peptide":
            model_name = "/root/peptide_embedding_model/"
            self.str_2_emb = str_2_emb_peptide
            self.action_list = get_peptide_all_general_action_list()
            self.drug_action_list = get_peptide_general_action_list
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif self.drug_type == "protein":
            model_name = "/root/protein_embedding_model/"
            self.str_2_emb = str_2_emb_protein
            self.action_list = get_protein_all_general_action_list()
            self.drug_action_list = get_protein_general_action_list
            self.model = BertModel.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
    
        if not torch.cuda.is_available():
            rl_model_dict = torch.load(self.rl_model_path, map_location=torch.device("cpu"))
        else:
            rl_model_dict = torch.load(self.rl_model_path)
        # CQL
        state_dim = rl_model_dict['state_dim']
        hidden_dim = rl_model_dict['hidden_dim']
        action_dim = rl_model_dict['action_dim']
        actor_lr = rl_model_dict['actor_lr']
        critic_lr = rl_model_dict['critic_lr']
        alpha_lr = rl_model_dict['alpha_lr']
        target_entropy = rl_model_dict['target_entropy']
        tau = rl_model_dict['tau']
        gamma = rl_model_dict['gamma']
        if torch.cuda.is_available():
            device = rl_model_dict['device']
        else:
            device = torch.device("cpu")
        rl_planner = RL_Planner(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device)
        rl_planner.load_state_dict(rl_model_dict['state_dict'])

        
        #PPO
        # state_dim = rl_model_dict['state_dim']
        # hidden_dim = rl_model_dict['hidden_dim']
        # action_dim = rl_model_dict['action_dim']
        # actor_lr = rl_model_dict['actor_lr']
        # critic_lr = rl_model_dict['critic_lr']
        # lmbda = rl_model_dict['lmbda']
        # epochs = rl_model_dict['epochs']
        # eps = rl_model_dict['eps']
        # gamma = rl_model_dict['gamma']
        # if torch.cuda.is_available():
        #     device = rl_model_dict['device']
        # else:
        #     device = torch.device("cpu")
        # rl_planner = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
        # rl_planner.load_state_dict(rl_model_dict['state_dict'])
        
        self.rl_planner = rl_planner
        
    
    @staticmethod
    def suggestion_to_actions(action_lists: list[str]) -> list[callable]:
        """Turn the suggestions returned by the planner model into actions"""
        actions = []
        for i, s in enumerate(action_lists):
            actions += [ActionAdder(s)]
        return actions
    
    def get_actions(
        self,
        states: list[ReasonerState],
        num_generate: int
    ) -> tuple[list[Callable], np.array]:
        attemps = 0
        action_priors = [None] * len(states)
        start = time.time()
        while any([i is None for i in action_priors]) and attemps < self.max_attempts:
            attemps += 1
            
            for i, s in enumerate(states):
                action_lists = []
                priors = []
                # root_mol = s.root_mol
                mol = s.best_mol
                mol_action_list = self.drug_action_list(mol)
                # mask_id = [i for i in range(len(self.action_list))]
                mask_id = [self.action_list.index(item) for item in mol_action_list]
                if mol in s.suggestion:
                    for prev_ac in s.suggestion[mol]:
                        mask_id.remove(self.action_list.index(prev_ac))

                # root_emb = np.array(str_2_emb([root_mol], self.smiles_tokenizer, self.smiles_model).squeeze().detach())
                # curr_emb = np.array(str_2_emb([mol], self.smiles_tokenizer, self.smiles_model).squeeze().detach())
                # state = np.concatenate((root_emb, curr_emb))
                #####################################################################################################
                state = np.array(self.str_2_emb([mol], self.tokenizer, self.model).squeeze().detach())

                for _ in range(num_generate):
                    # plain_action, prior = random.choice(self.action_list), 1.0
                    action = self.rl_planner.get_action(state, mask_id)
                    prior = 1.0
                    plain_action = self.action_list[action]

                    # print("*"*50)
                    # print(plain_action)
                    # vals = {
                    #     "query": s.generation_planner_prompt,
                    #     "suggestion": plain_action,
                    # }
                    # action = run_deepseek_prompts([fstr(refine_template, vals)], [system_prompt])[0]
                    action = plain_action
                    # try:
                    #     action = action["answer"].content
                    # except:
                    #     action = action["answer"]
                    # action = [] if action is None else parse_suggestion(action)
                    # try:
                    #     action = action[0]
                    # except:
                    #     action = plain_action
                    action_lists.append(action)
                    priors.append(prior)
                
                # if s.root:
                if False:
                    action_lists = [""] * len(action_lists)
                else:
                    for idx, ac in enumerate(action_lists):
                        print(f"RL Suggestion {idx}: {ac}", file=self.log_file)
                actions = self.suggestion_to_actions(action_lists)
                priors = np.array(priors)
                if s.valid_val == 0:
                    priors = np.array([0] * len(actions))
                action_priors[i] = (actions, priors)
        
        end = time.time()
        action_priors = [a_p if a_p is not None else [] for a_p in action_priors]
        
        return action_priors
                    
        
        