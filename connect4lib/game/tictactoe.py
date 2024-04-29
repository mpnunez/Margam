from connect4lib.utils import Connect4Exception
from connect4lib.game.game import Game
from connect4lib.transition import Transition
import numpy as np

class TicTacToe(Game):
    name = "TicTacToe"
    def __init__(self,nrows=3,ncols=3,nconnectwins=3):
        super().__init__(nrows,ncols,nconnectwins)
        self.options = list(range(nrows*ncols))

    def drop_in_slot(self, board, player: int, pos: int):
        board = board.copy()
        row = pos // self.nrows
        col = pos % self.nrows
        row_to_drop = self.nrows-1
        if np.sum(board[row,col,:]) == 1:
            return None
        board[row,col,player] = 1
        return board

    def get_legal_illegal_moves(self):
        legal_moves = []
        illegal_moves = []
        for r in range(self.nrows):
            for c in range(self.ncols):
                ind = self.nrows * r + c
                if np.sum(self.board[r,c,:]) == 0:
                    legal_moves.append(ind)
                else:
                    illegal_moves.append(ind)
        return legal_moves, illegal_moves
    
    def get_symmetric_transitions(self, tsn):

        row = tsn.selected_move // self.nrows
        col = tsn.selected_move % self.nrows
        yield tsn

        new_row = row
        new_col = (self.ncols-1) - col
        new_move = self.ncols*new_row+new_col
        yield Transition(
                board_state = tsn.board_state[:,::-1,:],
                selected_move = new_move,
                reward = tsn.reward,
                resulting_state = tsn.resulting_state[:,::-1,:],
            )

        new_row = (self.nrows-1) - row
        new_col = col
        new_move = self.ncols*new_row+new_col
        yield Transition(
                board_state = tsn.board_state[::-1,:,:],
                selected_move = new_move,
                reward = tsn.reward,
                resulting_state = tsn.resulting_state[::-1,:,:],
            )

        new_row = (self.nrows-1) - row
        new_col = (self.ncols-1) - col
        new_move = self.ncols*new_row+new_col
        yield Transition(
                board_state = tsn.board_state[::-1,::-1,:],
                selected_move = new_move,
                reward = tsn.reward,
                resulting_state = tsn.resulting_state[::-1,::-1,:],
            )

    