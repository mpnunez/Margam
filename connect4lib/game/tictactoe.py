from connect4lib.utils import Connect4Exception
from connect4lib.game.game import Game
import numpy as np

class TicTacToe(Game):
    name = "TicTacToe"
    def __init__(self,nrows=3,ncols=3,nconnectwins=3):
        super().__init__(nrows,ncols,nconnectwins)

    def drop_in_slot(self,player: int, pos: int):
        row = pos // self.nrows
        col = pos % self.nrows
        row_to_drop = self.nrows-1
        if self.board[player,row,col] == 1:
            raise Connect4Exception(f"No empty slot in row {row} column {col}")
        self.board[player,row,col] = 1

    def get_legal_illegal_moves(self):
        legal_moves = []
        illegal_moves = []
        for r in range(self.nrows):
            for c in range(self.ncols):
                ind = self.nrows * r + c
                if np.sum(self.board[:,r,c] == 0):
                    legal_moves.append(ind)
                else:
                    illegal_moves.append(ind)
        return legal_moves, illegal_moves

    def get_player_move(self,player,board_player_pov):
        """
        Get the move desired by the player

        If it is an illegal move, choose a random
        legal move for the player
        """
        
        legal_moves, _ = self.get_legal_illegal_moves()
        print(player.name)
        player_desired_move = player.get_move(board_player_pov,list(range(self.nrows*self.ncols)))
        if player_desired_move in legal_moves:
            return player_desired_move

        return random.choice(legal_moves)

    