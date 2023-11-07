from abc import ABC, abstractmethod
import numpy as np

class Player(ABC):
    def __init__(self,name=None):
        self.name = name or "nameless"
        
    @abstractmethod
    def get_move_scores(self,board: np.array) -> np.array:
        pass
    
class HumanPlayer(Player):
    def get_move_scores(self,board: np.array) -> np.array:
        n_cols = board.shape[2]
        invalid_input = True
        while invalid_input:
            new_input = input(f"Select column to drop token into [0-{n_cols-1}]\n")
            try:
                slot_to_drop = int(new_input)
            except ValueError:
                continue
            
            if 0 <= slot_to_drop and slot_to_drop < n_cols:
                invalid_input = False
                
        scores = np.zeros(n_cols)
        scores[slot_to_drop] = 1
        return scores
            
            
        
class RandomPlayer(Player):
    def get_move_scores(self,board: np.array) -> np.array:
        n_cols = board.shape[2]
        random_scores = np.random.rand(n_cols)
        random_scores = random_scores / random_scores.sum()
        return random_scores
        
    

