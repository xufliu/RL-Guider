"""Class for the llm planner policy"""
import traceback
import sys
import time
import re

from collections.abc import Callable

import numpy as np

sys.path.append("src")
from llm.prompt_template import priors_template
from search.state.molreasoner_state import ReasonerState
from search.policy.utils import (
    BabyPolicy, ActionAdder
)

# action_name_keys = {
#     "inclusion_criteria": IncludePropertyAdder,
#     "exclusion_criteria": ExcludePropertyAdder,
#     "relationship_to_candidate_list": RelationToCandidateListChanger,
# }

def remove_parentheses_content(s):
    count = 0
    while '(' in s:
        count += 1
        if count >= 10:
            return s
        s = re.sub(r'\([^()]*\)', '', s)
    return s

class LLM_Planner_Policy(BabyPolicy):
    """A policy using LLM as Planner."""
    def __init__(
        self,
        log_file,
        llm_function: callable = lambda list_x: [example_output] * len(list_x),
        max_attempts: int=3,
        log_path: str="",
    ):
        self.llm_function = llm_function
        self.max_attempts = max_attempts
        self.log_path = log_path
        self.log_file = log_file
        
    @staticmethod
    def suggestion_to_actions(action_lists: list[str]) -> list[callable]:
        """Turn the suggestions returned by the planner model into actions"""
        actions = []
        for i, s in enumerate(action_lists):
            actions += [ActionAdder(s)]
        return actions
    
    """
    input: leaves of the search tree (multiple states)
    output: for each state, return its actions and the corresponding priors
    """
    def get_actions(
        self,
        states: list[ReasonerState],
        num_generate: int,
    ) -> tuple[list[Callable], np.array]:
        attempts = 0
        action_priors = [None] * len(states)
        start = time.time()
        llm_answers = None
        while any([i is None for i in action_priors]) and attempts < self.max_attempts and llm_answers is None:
            attempts += 1
            prompts = []
            prompts_idx = []
            
            is_root = False
            for i, s in enumerate(states):
                if s.root:
                    is_root = True
                messages = []
                messages.append({"role": "system", "content": s.generation_system_prompt})
                if s.priors_template is None:
                    s.priors_template = priors_template
                messages.append({"role": "user", "content": s.priors_prompt(num_generate)})
                prompts_idx.append(i)
                prompts.append(messages)
                print(f"AI Planning Prompt {i}: {messages}", file=self.log_file)

            if len(prompts) > 0:
                if True:
                # if not is_root:
                    try:
                        llm_answers = self.llm_function(prompts)
                    except Exception:
                        print("llm process failure.", file=self.log_file)
                        continue
                    for i, ans in enumerate(llm_answers):
                        try:
                            prior_answer = ans["answer"]["content"]
                        except:
                            prior_answer = ans["answer"]
                        # if isinstance(ans, str):
                        #     prior_answer = ans["answer"]["content"]
                        # else:
                        #     prior_answer = ans["answer"]
                        print(f"AI Suggestion {i}: {prior_answer}", file=self.log_file)
                        
                        try:
                            s = states[prompts_idx[i]]
                            action_lists = s.process_prior(prior_answer)
                            action_lists = [remove_parentheses_content(a) for a in action_lists]
                            print(action_lists)
                            actions = self.suggestion_to_actions(action_lists) # new node generated!

                            if len(actions) >= num_generate:
                                actions = actions[: num_generate]
                                priors = np.array([1 / len(actions)] * len(actions)) # score for action priority
                            elif len(actions) < num_generate:
                                length_difference = num_generate - len(actions)
                                priors = np.array(
                                    [1 / len(actions)] * len(actions)
                                    + [0.1] * length_difference
                                )
                                # actions += [None] * length_difference
                                actions += [actions[0]] * length_difference

                            action_priors[prompts_idx[i]] = (actions, priors)
                        except Exception as e:
                            print(f"Could not parse the actions for the given state. Error: {e}", file=self.log_file)
                else:
                    action_lists = [""]
                    for i in range(len(prompts)):
                        actions = self.suggestion_to_actions(action_lists) # new node generated!
                        if len(actions) >= num_generate:
                            actions = actions[: num_generate]
                            priors = np.array([1 / len(actions)] * len(actions)) # score for action priority
                        elif len(actions) < num_generate:
                            length_difference = num_generate - len(actions)
                            priors = np.array(
                                [1 / len(actions)] * len(actions)
                                + [0.1] * length_difference
                            )
                            # actions += [None] * length_difference
                            actions += [actions[0]] * length_difference

                        action_priors[prompts_idx[i]] = (actions, priors)
                
        end = time.time()
        action_priors = [a_p if a_p is not None else [] for a_p in action_priors]
        
        return action_priors