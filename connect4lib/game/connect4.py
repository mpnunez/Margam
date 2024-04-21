from connect4lib.utils import Connect4Exception
from connect4lib.game.game import Game

class Connect4(Game):
    name = "Connect4"
    def __init__(self,nrows=3,ncols=3,nconnectwins=3):
        super().__init__(nrows,ncols,nconnectwins)
        self.options = list(range(ncols))

    def drop_in_slot(self,board, player: int, pos: int):
        board = board.copy()
        row_to_drop = self.nrows-1
        while row_to_drop>=0:
            if np.sum(board[row_to_drop,col,:]) == 0:
                board[row_to_drop,col,player] = 1
                return board
            row_to_drop -= 1
        
        return None

    def get_legal_illegal_moves(self):
         return [i for i in range(self.ncols) if np.sum(self.board[0,i,:]) == 0],
            [i for i in range(self.ncols) if np.sum(self.board[0,i,:]) > 0]
