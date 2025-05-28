"""Implement Search Tree"""
from pathlib import Path
import pickle
import sys
import time

from random import shuffle

from typing import TypeVar

import numpy as np

SearchTree = TypeVar("SearchTree")

sys.path.append("src")
from search.reward.reward_function import reward_function

def init_search_tree(args, data, llm_function, policy, log_file):
    """Get the search tree provided in args."""
    assert isinstance(args.num_keep, int) and args.num_keep > 0, "invalid parameter"
    assert (
            isinstance(args.num_generate, int) and args.num_generate > 0
        ), "invalid parameter"
    return SearchTree(
        data,
        llm_function,
        policy,
        num_generate=args.num_generate,
        num_keep=args.num_keep,
        log_file=log_file
    )


class SearchTree:
    def __init__(
        self, data, llm_function, policy, num_generate, num_keep, log_file, root_reward=False
    ):
        """Creare a SearchTree from root node."""
        self.num_generate = num_generate
        self.num_keep = num_keep
        self.llm_function = llm_function
        self.policy = policy
        self.nodes = []
        self.nodes.append([data])
        self.parent_idx = [[-1]]
        self.reward_fn = reward_function
        if root_reward:
            self.node_rewards = [[data.reward]]
        else:
            self.node_rewards = [[0]]
        self.generated_nodes = [[]]
        self.generated_node_rewards = [[]]
        self.generated_parent_idx = [[]]
        
        self.start_time = None
        self.end_time = None
        
        self.best_node = None
        self.best_node_prop = None
        self.best_reward = 0
        self.log_file = log_file
        
    def expand_node(self, nodes):
        # nodes is a list of ReasonerState
        actions_priors = self.policy.get_actions(nodes, self.num_generate)
        new_nodes = []
        parent_idx = []
        for i, node in enumerate(nodes):
            actions, these_priors = actions_priors[i]
            shuffle_idx = list(range(len(these_priors)))
            shuffle(shuffle_idx)
            these_priors = [these_priors[i] for i in shuffle_idx]
            actions = [actions[i] for i in shuffle_idx]
            
            action_idxs = np.argsort(these_priors)[-self.num_generate :]
            these_new_nodes = []
            for j in action_idxs:
                if these_priors[j] > 0:
                    a = actions[j]
                    new_node = a(node)
                    # print("new node messages: ", new_node.messages)
                    these_new_nodes.append(new_node) # a(node) is a new node (ReasonerState)
            self.run_generation(these_new_nodes)
            new_nodes += these_new_nodes
            parent_idx += [i] * len(these_new_nodes)
        return new_nodes, parent_idx
        
    def run_generation(self, nodes):
        # nodes is a list of ReasonerState
        attempts = 0
        max_attempts = 3
        # message list
        messages_list = []
        for i, s in enumerate(nodes):
            messages = s.generation_prompt
            print(f"Message {i}: {messages}", file=self.log_file)
            messages_list.append(messages)
        generation_results = None
        print("Message: ", messages_list)
        
        try:
            while generation_results is None and attempts < max_attempts:
                attempts += 1
                generation_results = self.llm_function(messages_list)
        except Exception as err:
            print("ERROR: Generating results error.", file=self.log_file)
            raise err
            
        print("Generation Result: ", generation_results)
        for i, s in enumerate(nodes):
            try:
                answer_txt = generation_results[i]["answer"]["content"]
            except:
                answer_txt = generation_results[i]["answer"]
            print(f"AI answer {i}: {answer_txt}", file=self.log_file)
            s.process_generation(results=generation_results[i])
            if s.valid_val == 1:
                reward = self.reward_fn([s])[0]
                if reward > self.best_reward:
                    self.best_node = s
                    self.best_node_prop = s.prop
    
                    
    def simulation_policy(self):
        """Simulate a beam search step."""
        answer = "valid"
        if self.start_time is None:
            self.start_timer()
            
        # expand final layer of nodes
        last_layer_nodes = self.nodes[-1]
        expanded_nodes = []
        invalid = True
        for i, s in enumerate(last_layer_nodes):
            if len(s.prev_mol_list) != len(set(s.prev_mol_list)):
                answer = "duplicate"
                invalid = False
            elif s.valid_val == 1 :
                expanded_nodes.append(s)
                invalid = False
        if invalid:
            print("All nodes in last layer is invalid, Tree search process ends.", file=self.log_file)
            answer = "invalid"
            return answer
        successor_nodes, parent_idx = self.expand_node(expanded_nodes)
        
        try:
            # calculate their rewards
            successor_rewards = self.reward_fn(successor_nodes)
        except Exception as err:
            print("ERROR:Reward function call failed. Returning a penalty value.", file=self.log_file)
            successor_rewards = [-10] * len(successor_nodes)
            raise err
        
        # selected node index
        selected_node_idx = np.argsort(successor_rewards)[
            -self.num_keep :
        ]
        print("successor rewards: ", successor_rewards)
        generated_idx = np.argsort(successor_rewards)[
            : -self.num_keep
        ]

        print(selected_node_idx)
        # seperate out the top-k rewards
        selected_nodes = [successor_nodes[i] for i in selected_node_idx]
        selected_rewards = [successor_rewards[i] for i in selected_node_idx]
        selected_parents = [parent_idx[i] for i in selected_node_idx]
        
        # seperate out the other nodes that were not chosen (generated_nodes)
        generated_nodes = [successor_nodes[i] for i in generated_idx]
        generated_node_rewards = [successor_rewards[i] for i in generated_idx]
        generated_parent_idx = [parent_idx[i] for i in generated_idx]
        
        # save selected nodes
        self.nodes.append(selected_nodes)
        self.node_rewards.append(selected_rewards)
        self.parent_idx.append(selected_parents)
        
        # Save the generated_nodes
        self.generated_nodes.append(generated_nodes)
        self.generated_node_rewards.append(generated_node_rewards)
        self.generated_parent_idx.append(generated_parent_idx)
        
        return answer
    
    def start_timer(self):
        """Save the time to the start time."""
        self.start_time = time.time()
    
    def end_timer(self):
        """Save a number to the end timer."""
        self.end_time = time.time()
    
    def get_time(self):
        """Save a number to the end timer."""
        return self.end_time - self.start_time
    
    def reset_timer(self):
        """Reset the time values to None."""
        self.start_time = None
        self.end_time = None

    def get_processed_data(self) -> dict:
        """Turn beam search tree into dictionary for saving."""
        beam_search_data = dict()
        beam_search_data['nodes'] = []
        for list_nodes in self.nodes:
            beam_search_data['nodes'].append([vars(n) for n in list_nodes])
        beam_search_data["node_rewards"] = self.node_rewards
        beam_search_data["parent_idx"] = self.parent_idx
        
        beam_search_data["generated_nodes"] = []
        for list_nodes in self.generated_nodes:
            beam_search_data["generated_nodes"].append([vars(n) for n in list_nodes])
        beam_search_data["generated_node_rewards"] = self.generated_node_rewards
        beam_search_data["generated_parent_idx"] = self.generated_parent_idx
        
        beam_search_data["num_generate"] = self.num_generate
        beam_search_data["num_keep"] = self.num_keep

        beam_search_data["start_time"] = self.start_time
        beam_search_data["end_time"] = self.end_time

        return beam_search_data
    
    @classmethod
    @staticmethod
    def from_data(search_data: dict, llm_function, policy, log_file, node_constructor=None):
        """Create a beam search object from stored data."""
        new_tree = SearchTree(
            None, llm_function, policy, None, None, log_file=log_file, root_reward=False
        )
        
        for i, list_nodes in enumerate(search_data["nodes"]):
            new_nodes = [node_constructor(n) for n in list_nodes]
            if i < len(new_tree.generated_nodes):
                new_tree.nodes[i] = new_nodes
            else:
                new_tree.nodes.append(new_nodes)
                
        
        new_tree.node_rewards = search_data['node_rewards']
        new_tree.parent_idx = search_data['parent_idx']
        
        
        for i, list_nodes in enumerate(search_data["generated_nodes"]):
            new_nodes = [node_constructor(n) for n in list_nodes]
            if i < len(new_tree.generated_nodes):
                new_tree.generated_nodes[i] = new_nodes
            else:
                new_tree.generated_nodes.append(new_nodes)
        
        new_tree.generated_node_rewards = search_data['generated_node_rewards']
        new_tree.parent_idx = search_data['parent_idx']
        
        new_tree.num_generate = search_data['num_generate']
        new_tree.num_keep = search_data['num_keep']
        
        new_tree.start_time = search_data["start_time"]
        new_tree.end_time = search_data["end_time"]
        return new_tree
    
    def pickle(self, fname: Path):
        """Save beam search to pickle file."""
        pickle_data = self.get_processed_data()
        with open(fname, 'wb') as f:
            pickle.dump(pickle_data, f)
        
    def step_return(self):
        """Take a step and return the tree data."""
        answer = self.simulation_policy()
        self.end_timer()
        return self.get_processed_data(), answer
        
    def step_save(self, fname):
        """Take a simulation step and save the resulting tree state with end_time."""
        answer = self.simulation_policy()
        self.end_timer()
        self.pickle(fname)
        
    def __len__(self):
        """Return the depth of self."""
        return len(self.nodes) - 1