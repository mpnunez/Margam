from enum import Enum
import numpy as np

"""
Connect 4
N rows
M columns
C 
"""

class Connect4Exception(Exception):
    pass

class Color(Enum):
    BLUE = 0
    RED = 1

class Game:
    def __init__(self,nrows=6,ncols=7,nconnectwins=4):
        self.nrows = nrows
        self.ncols = ncols
        self.nconnectwins = nconnectwins
        self.nplayers = 2
        self.board = np.zeros([self.nrows,self.ncols,self.nplayers])
        self.current_player = 0

    def check_horizontal_win(self,player: int) -> bool:
        for row in range(self.nrows):
            n_consecutive = 0
            for col in range(self.ncols):
                if self.board[row,col,player] == 1:
                    n_consecutive += 1
                else:
                    n_consecutive = 0
                if n_consecutive == self.nconnectwins:
                    return True
            
        return False
    
    def check_vertical_win(self,player: int) -> bool:
        for col in range(self.ncols):
            n_consecutive = 0
            for row in range(self.nrows):
                if self.board[row,col,player] == 1:
                    n_consecutive += 1
                else:
                    n_consecutive = 0
                if n_consecutive == self.nconnectwins:
                    return True
            
        return False
    
    def check_diagonal_ll_ur_win(self,player: int) -> bool:
        """
        Sum of row and col is constant
        """
        for row_col_sum in range(self.nrows+self.ncols-1):
            n_consecutive = 0
            imin = np.max([0,row_col_sum-self.ncols+1])
            imax = np.min([row_col_sum,self.nrows-1])
            for row in range(imin,imax+1):
                col = row_col_sum - row
                if self.board[row,col,player] == 1:
                    n_consecutive += 1
                else:
                    n_consecutive = 0
                if n_consecutive == self.nconnectwins:
                    return True
            
        return False
    
    def check_diagonal_ul_lr_win(self,player: int) -> bool:
        """
        row# - col# is constant
        """
        for row_col_diff in range(-self.ncols,self.nrows+1):
            n_consecutive = 0
            imin = np.max([0,row_col_diff])
            imax = np.min([self.ncols+row_col_diff-1,self.nrows-1])
            for row in range(imin,imax+1):
                col = row - row_col_diff
                if self.board[row,col,player] == 1:
                    n_consecutive += 1
                else:
                    n_consecutive = 0
                if n_consecutive == self.nconnectwins:
                    return True
            
        return False


    def check_win(self,player: int = 0) -> bool:
        """
        Return True if there are 4 1's in a row in the first channel
        """
        
        return self.check_horizontal_win(player) or self.check_vertical_win(player) or self.check_diagonal_ll_ur_win(player) or self.check_diagonal_ul_lr_win(player)
    
    def drop_in_slot(self,player: int, col: int):
        row_to_drop = self.nrows-1
        while row_to_drop>=0:
            if np.sum(self.board[row_to_drop,col,:]) == 0:
                self.board[row_to_drop,col,player] = 1
                return
            row_to_drop -= 1
            
        raise Connect4Exception(f"No empty slot in column {col}")
    
    def show_board(self):
        print(self.board)




    
"""
Game position is a numpy array of size
nrows x ncols x nplayers
"""
# Game position is a 
# Position is a np array
# Current player is always

