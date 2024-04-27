from connect4lib.utils import Connect4Exception
from connect4lib.game.game import Game

import numpy as np
from typing import List

class Connect4(Game):
    name = "Connect4"
    def __init__(self,nrows=6,ncols=7,nconnectwins=4):
        super().__init__(nrows,ncols,nconnectwins)
        self.options = list(range(ncols))

    def drop_in_slot(self,board, player: int, pos: int):
        board = board.copy()
        row_to_drop = self.nrows-1
        while row_to_drop>=0:
            if np.sum(board[row_to_drop,pos,:]) == 0:
                board[row_to_drop,pos,player] = 1
                return board
            row_to_drop -= 1
        
        return None

    def get_legal_illegal_moves(self):
         return [i for i in range(self.ncols) if np.sum(self.board[0,i,:]) == 0], [i for i in range(self.ncols) if np.sum(self.board[0,i,:]) > 0]

    def get_symmetric_transitions(self, tsn):
        yield tsn
        yield Transition(
                board_state = tsn.board_state[:,::-1,:],
                selected_move = (self.ncols-1) - tsn.selected_move,
                reward = tsn.reward,
                resulting_state = tsn.resulting_state[:,::-1,:],
            )
