"""Class for the base policy (without planner)"""
import traceback
import sys
import time

from typing import List, Tuple
from collections.abc import Callable

import numpy as np

sys.path.append("src")
from search.state.molreasoner_state import ReasonerState
from search.policy.utils import (
    BabyPolicy, ActionAdder
)


class Base_Policy(BabyPolicy):
    """A basic policy"""
    def __init__(
        self,
        log_file,
        llm_function: callable = lambda list_x: [example_output] * len(list_x),
        max_attempts: int=3,
        log_path: str="",
    ):
        self.log_file = log_file
        self.llm_function = llm_function
        self.max_attempts = max_attempts
        self.log_path = log_path
    
    @staticmethod
    def suggestion_to_actions(action_lists: list[str]) -> list[callable]:
    # def suggestion_to_actions(action_lists: List[str]) -> List[callable]:
    
        """Turn the suggestions returned by the planner model into actions"""
        actions = []
        for i, s in enumerate(action_lists):
            actions += [ActionAdder(s)]
        return actions
    
        
    def get_actions(
        self,
        states: list[ReasonerState],
        # states: List[ReasonerState],
        
        num_generate: int
    ) -> tuple[list[Callable], np.array]:
    # ) -> Tuple[List[Callable], np.ndarray]:
    
        attempts = 0
        action_priors = [None] * len(states)
        start = time.time()
        while any([i is None for i in action_priors]) and attempts < self.max_attempts:
            attempts += 1
            
            for i, s in enumerate(states):
                action_lists = [""] * num_generate
                actions = self.suggestion_to_actions(action_lists)
                if s.valid_val == 1:
                    priors = np.array([1 / len(actions)] * len(actions))
                else:
                    priors = np.array([0] * len(actions))
                action_priors[i] = (actions, priors)
            
        end = time.time()
        action_priors = [a_p if a_p is not None else [] for a_p in action_priors]
        
        return action_priors
                