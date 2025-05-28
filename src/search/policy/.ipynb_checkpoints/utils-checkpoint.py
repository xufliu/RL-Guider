"""A base class for policies"""
from abc import ABCMeta, abstractmethod
from collections.abc import Callable

import numpy as np
from typing import List, Tuple
from copy import deepcopy
        
class BabyPolicy(metaclass=ABCMeta):
    """A base class for policies."""

    @abstractmethod
    def get_actions(
        self, states: list[object]
        # self, states: List[object]
        
    ) -> tuple[list[Callable], list[np.array]]:
    # ) -> Tuple[List[Callable], List[np.ndarray]]:
        
        """Return the actions along with their priors."""
        ...


class ActionAdder:
    """Class to add action to a state"""
    
    def __init__(self, suggestion):
        self.suggestion = suggestion
        if self.suggestion != "":
            # self.prefix = "You are suggested to do the modification according to the suggestion: "
            self.prefix = ""
        else:
            self.prefix = ""
        self._message = (
            f"{self.prefix}{self.suggestion}"
        )
    
    def __call__(self, state, trial=False):
        """Add property to the state"""
        new_state = state.return_next()
        new_state.suggestion = deepcopy(state.suggestion)
        if self._message != "":
            new_state.suggestion[state.best_mol].append(self._message)
        if not trial:
            pass
        return new_state
    
    def message(self, state):
        """Return the message for this action. State does nothing."""
        return self._message