import numpy as np
from utils import Connect4Exception
from move_record import MoveRecord


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
        print()
        
    def get_valid_invalid_moves(self):
        return [i for i in range(self.ncols) if np.sum(self.board[:,0,i]) == 0], [i for i in range(self.ncols) if np.sum(self.board[:,0,i]) > 0]

    def get_player_move(self,player,board_player_pov):
        """
        Get the legal move the player evaluates as highest
        """
        
        # Get all move scores
        player_move_scores = player.get_move_scores(board_player_pov)
        scored_moves = [(-score,ind) for ind, score in enumerate(player_move_scores)]
        scored_moves = sorted(scored_moves)
        
        # Get the valid move with the highest player score
        valid_moves, _ = self.get_valid_invalid_moves()
        for _, col in scored_moves:
            if col in valid_moves:
                actual_move = col
                break
            
        return actual_move
        
        
    def play_game(self,show_board_each_move=False,verbose=False):
        
        if len(self.players) != self.nplayers:
            raise Connect4Exception(f"Need {self.nplayers} players")
        
        game_data = []
        
        continue_playing = True
        winner = -1
        move_ind = -1
        while continue_playing:
        
            for ind, player in enumerate(self.players):
                move_ind += 1
                legal_moves, illegal_moves = self.get_valid_invalid_moves()
                if len(legal_moves) == 0:
                    continue_playing = False
                    break
                
                if show_board_each_move:
                    self.show_board()
                    
                board_player_pov = self.board if ind == 0 else self.board[::-1,:,:]
                player_move = self.get_player_move(player,board_player_pov)
                
                move_record = MoveRecord(
                    board_state = board_player_pov.copy(),
                    legal_moves = legal_moves,
                    illegal_moves = illegal_moves,
                    selected_move = player_move,
                    move_ind = move_ind,
                    )
                game_data.append(move_record)
                    
                self.drop_in_slot(ind,player_move)
                if self.check_win(ind):
                    
                    if show_board_each_move:
                        self.show_board()
                    winner = ind
                    continue_playing = False
                    break
                
        
        for ind, mr in enumerate(game_data):
            mr.game_length = move_ind
            mr.result = 0.5 if winner == -1 else int(ind % self.nplayers == winner)
        
        if verbose:
            if winner == -1:
                print("Game was a draw")
            else:
                print(f"Player {self.players[winner]} won!")
            
        return winner, game_data
