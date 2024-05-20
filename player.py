from abc import ABC, abstractmethod
import numpy as np
import random
from typing import List

from collections import defaultdict
import numpy as np
import random

from typing import Tuple, Optional


from utils import get_training_and_viewing_state

class Player(ABC):
    requires_user_input = False
    
    def __init__(self,name=None):
        self.name = name or "nameless"

    @abstractmethod
    def get_move(self, game, state) -> int:
        pass
    
    
class HumanPlayer(Player):
    requires_user_input = True
    def get_move(self,game,state) -> int:
        
        valid_input = False
        while not valid_input:
            print("\nBoard state")
            state_np_for_cov, human_view_state = get_training_and_viewing_state(game,state)
            print("Available moves:")
            print(state.legal_actions())
            new_input = input(f"Select a move:")
            
            try:
                move_to_play = int(new_input)
            except ValueError:
                continue
            
            valid_input = move_to_play in state.legal_actions()
                
        return move_to_play
        
class RandomPlayer(Player):
    def get_move(self,game,state) -> int:
        return random.choice(state.legal_actions())
        
class ColumnSpammer(Player):
    def __init__(self,name=None,move_preference=0):
        super().__init__(name)
        self.favorite_move = move_preference
        
    def get_move(self,game,state) -> int:
        if self.favorite_move in state.legal_actions():
            return self.favorite_move
        return random.choice(state.legal_actions())



class MiniMax(Player):
    """
    Only works for 2 player games

    Depth 0: random player
    Depth 1: Always makes winning move if available
    Depth 2: Blocks opponent from winning on next move
    Depth 3: Sets up forced win on next move
    etc.
    """
    
    def __init__(self,*args,max_depth=3,**kwargs):
        super().__init__(*args,**kwargs)
        self.max_depth = max_depth

    def eval_state(
        self,
        board: np.array,
        game,
        depth=1,
        current_player=0) -> Tuple[float, Optional[int]]:
        """
        Returns a tuple with
        - The value of the current board for player 0
        - The best move to be taken for current agent
        """

        if game.check_win(board,0):
            return (game.WIN_REWARD, None)
        if game.check_win(board,1):
            return (game.LOSS_REWARD, None)
        if depth <= 0:
            return (game.TIE_REWARD, random.choice(game.options))      # Neither player can force a win

        move_values = defaultdict(list)
        for move in game.options:
            board_result = game.drop_in_slot(board,current_player,move)
            if board_result is None:
                continue
            
            value, _ = self.eval_state(board_result, game, depth-1, 1 - current_player)
            move_values[value].append(move)

        if len(move_values) == 0:
            return (game.TIE_REWARD,None)

        if current_player == 0:
            move_value = max(move_values.keys())
        else:
            move_value = min(move_values.keys())
        return move_value, random.choice(move_values[move_value])

    def get_move(self, board: np.array, game) -> int:
        value, move = self.eval_state(board,game,depth=self.max_depth)
        return move
