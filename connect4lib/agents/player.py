from abc import ABC, abstractmethod
import numpy as np
import random
from typing import List

class Player(ABC):
    requires_user_input = False
    
    def __init__(self,name=None):
        self.name = name or "nameless"

    @abstractmethod
    def get_move(self, board: np.array, game) -> int:
        pass
    
    
class HumanPlayer(Player):
    requires_user_input = True
    def get_move(self,board: np.array, game) -> int:
        
        valid_input = False
        while not valid_input:
            print("\nAvailable moves:")
            print(game.options)
            new_input = input(f"Select a move:")
            
            try:
                slot_to_drop = int(new_input)
            except ValueError:
                continue
            
            valid_input = slot_to_drop in game.options
                
        return slot_to_drop
        
class RandomPlayer(Player):
    def get_move(self,board: np.array, game) -> int:
        return random.choice(game.options)
        
class ColumnSpammer(Player):
    def __init__(self,name=None,col_preference=0):
        super().__init__(name)
        self.favorite_column = col_preference
        
    def get_move(self,board: np.array, game) -> int:
        return self.favorite_column
