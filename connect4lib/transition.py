from dataclasses import dataclass
import numpy as np

@dataclass
class Transition:
    """
    Data from the game used to train the agent
    """
    state: np.ndarray = None
    selected_move: int = 0
    reward: float = 0
    next_state: np.ndarray = None
