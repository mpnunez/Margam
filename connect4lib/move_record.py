from dataclasses import dataclass
import numpy as np

@dataclass
class MoveRecord:
    """
    Data from the game used to train the agent
    """
    board_state: np.ndarray = None
    selected_move: int = 0
    reward: int = 0
    resulting_state: np.ndarray = None
