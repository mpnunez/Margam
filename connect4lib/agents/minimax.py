from collections import defaultdict
from connect4lib.agents.player import Player
import numpy as np
import random

from typing import Tuple, Optional

import copy

class MiniMax(Player):
    """
    Only works for 2 player games
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

        # Need to implement
        # 1. Detecting if player 0 or 1 has won
        # 2. Modifying the board when current player does BLANK move

        # Check if either player has won
        if game.check_win(board,0):
            return (game.WIN_REWARD, None)
        if game.check_win(board,1):
            return (game.LOSS_REWARD, None)
        if depth <= 0:
            return (game.TIE_REWARD, None)      # Neither player can force a win

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
