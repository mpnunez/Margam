import numpy as np
from utils import Connect4Exception

"""
Connect 4
N rows
M columns
C 
"""



class Game:
    def __init__(self,nrows=6,ncols=7,nconnectwins=4):
        self.nrows = nrows
        self.ncols = ncols
        self.nconnectwins = nconnectwins
        self.nplayers = 2
        self.board = np.zeros([self.nplayers,self.nrows,self.ncols])
        self.current_player = 0
        self.players = []

    def check_horizontal_win(self,player: int) -> bool:
        for row in range(self.nrows):
            n_consecutive = 0
            for col in range(self.ncols):
                if self.board[player,row,col] == 1:
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
                if self.board[player,row,col] == 1:
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
                if self.board[player,row,col] == 1:
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
                if self.board[player,row,col] == 1:
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
            if np.sum(self.board[:,row_to_drop,col]) == 0:
                self.board[player,row_to_drop,col] = 1
                return
            row_to_drop -= 1
            
        raise Connect4Exception(f"No empty slot in column {col}")
    
    def show_board(self):
        display_board = self.board[0,:,:] + 2 * self.board[1,:,:]
        print(display_board)
        
    def get_valid_moves(self):
        return [i for i in range(self.ncols) if np.sum(self.board[:,0,i]) == 0]

    def let_player_move(self,player:int):
        """
        Return true if the player just won
        """
        
        # Need to communicate to the player which # player they are
        
        player_move_scores = self.players[player].get_move_scores(self.board,player)
        scored_moved = [(-score,ind) for ind, score in enumerate(player_move_scores)]
        scored_moved = sorted(scored_moved)
        
        # Get the valid move with the highest player score
        valid_moves = self.get_valid_moves()
        for _, col in scored_moved:
            if col in valid_moves:
                actual_move = col
                break
            
        self.drop_in_slot(player,actual_move)
        
        # Check for the player's win
        
    def play_game(self,show_board_each_move=False):
        while len(self.get_valid_moves())>0:
        
            for ind, _ in enumerate(self.players):
                
                if show_board_each_move:
                    self.show_board()
                self.let_player_move(ind)
                if self.check_win(ind):
                    print(f"Player {ind} won!")
                    if show_board_each_move:
                        self.show_board()
                    return
                
            
            
        print("Game was a draw")
        
        
            
        


    
"""
Game position is a numpy array of size
nrows x ncols x nplayers
"""
# Game position is a 
# Position is a np array
# Current player is always

