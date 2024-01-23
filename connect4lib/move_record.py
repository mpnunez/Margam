from dataclasses import dataclass
import numpy as np

@dataclass
class MoveRecord:
    # Data from the game
    board_state: np.ndarray = np.zeros([2,6,7])
    legal_moves: list = None
    illegal_moves: list = None
    selected_move: int = 0
    move_ind: int = 0
    game_length: int = 0
    player_name: str = ""
    
    # Results
    result: int = 0
    move_scores: np.ndarray = np.zeros(7)

    def __lt__(self,other):
        self_score = (self.result, self.move_ind-self.game_length, 1 / (1+self.game_length))
        other_score = (other.result, other.move_ind-other.game_length, 1 / (1+self.game_length))
        return self_score < other_score
