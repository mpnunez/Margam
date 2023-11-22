from abc import ABC, abstractmethod
import numpy as np
import random

class Player(ABC):
    def __init__(self,name=None,randomness_weight=0):
        self.name = name or "nameless"
        self.random_weight = randomness_weight
        
    def get_random_move_scores(self,board: np.array) -> np.array:
        n_cols = board.shape[2]
        random_scores = np.random.rand(n_cols)
        random_scores = random_scores / random_scores.sum()
        return random_scores
        
    @abstractmethod
    def get_move_scores_deterministic(self,board: np.array) -> np.array:
        pass
    
    def get_move_scores(self,board: np.array) -> np.array:
        return self.random_weight * self.get_random_move_scores(board) + (1-self.random_weight)*self.get_move_scores_deterministic(board)
    
class HumanPlayer(Player):
    def get_move_scores_deterministic(self,board: np.array) -> np.array:
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
    def get_move_scores_deterministic(self,board: np.array) -> np.array:
        return self.get_random_move_scores(board)
        
class ColumnSpammer(Player):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.randomize_column_preferences()
        
    def randomize_column_preferences(self):
        self.col_preferences = list(range(7))
        random.shuffle(self.col_preferences)
        self.col_preferences = np.array(self.col_preferences)
        self.col_preferences = self.col_preferences / self.col_preferences.sum()
        
    def get_move_scores_deterministic(self,board: np.array) -> np.array:
        # Change preferences if at the beginning of a game
        if board.sum() < 2:
            self.randomize_column_preferences()
        return np.array(self.col_preferences)
    

