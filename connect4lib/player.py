from abc import ABC, abstractmethod
import numpy as np
import random

class Player(ABC):
    requires_user_input = False
    
    def __init__(self,name=None):
        self.name = name or "nameless"

    @abstractmethod
    def get_move(self,board: np.array) -> int:
        pass
    
    
class HumanPlayer(Player):
    requires_user_input = True
    def get_move(self,board: np.array) -> np.array:
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
                
        return slot_to_drop
            
            
        
class RandomPlayer(Player):
    def get_move(self,board: np.array) -> np.array:
        n_cols = board.shape[2]
        return random.choice(range(n_cols))
        
class ColumnSpammer(Player):
    def __init__(self,name=None,col_preference=0):
        super().__init__(name)
        self.favorite_column = col_preference
        
    def get_move(self,board: np.array) -> int:
        return self.favorite_column
