from connect4lib.agents.player import Player
import numpy as np
import random

from typing import Tuple

import copy

class PolicyPlayer(Player):
    
    def eval_state(state,depth=3):

        # max of moves we can do from here

        if is_winning(state):
            return 1
        if is_losing(state):
            return -1
        
        legal_moves = get_legal_moves()
        next_states = {i: transform_state(board_move)
            for i in legal_moves}

        possible_results = []
        for move in range(7):
            state = board.transform(move)
            resulting_boards = []
            for op_move in range(7):
                state = board.transform(op_move)
                value = eval_state(state,depth-1)
                resulting_boards.append((value, state))
            worst_val, worst_next_state = min(resulting_boards)
            possible_results.append(worst_val, worst_next_state)
        best_val, best_next_state = max(possible_results)
                

        if depth == 0:


    
    def get_move(self,board: np.array, options: List[int]) -> int:
        
        # Choose best move through recursion

        return 0
