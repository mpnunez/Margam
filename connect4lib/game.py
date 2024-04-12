import numpy as np
from connect4lib.utils import Connect4Exception
from connect4lib.transition import Transition
from enum import Enum
import random

class GameStatus(Enum):
    NOTSTARTED = 1
    INPROGRESS = 2
    COMPLETE = 3

class Game:
    def __init__(self,nrows=6,ncols=7,nconnectwins=4):
        self.nrows = nrows
        self.ncols = ncols
        self.nconnectwins = nconnectwins
        self.nplayers = 2
        self.board = np.zeros([self.nplayers,self.nrows,self.ncols])
        self.current_player_ind = 0
        self.players = []
        self.status = GameStatus.NOTSTARTED
        self.game_data = []
        self.winner = -1
        self.move_ind = 0
        self.verbose = False

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
        display_board = sum((i+1)*self.board[i,:,:] for i in range(len(self.board)))
        print(display_board)
        print()
        
    def get_legal_illegal_moves(self):
        return [i for i in range(self.ncols) if np.sum(self.board[:,0,i]) == 0], [i for i in range(self.ncols) if np.sum(self.board[:,0,i]) > 0]

    def get_player_move(self,player,board_player_pov):
        """
        Get the legal move the player evaluates as highest
        """
        
        legal_moves, _ = self.get_legal_illegal_moves()
        player_desired_move = player.get_move(board_player_pov)
        if player_desired_move in legal_moves:
            return player_desired_move

        return random.choice(legal_moves)

        
    
    def start_game(self):
        if len(self.players) != self.nplayers:
            raise Connect4Exception(f"Need {self.nplayers} players")
        self.board = np.zeros([self.nplayers,self.nrows,self.ncols])
        self.status = GameStatus.INPROGRESS
    
    def get_next_player_move(self):
        player = self.players[self.current_player_ind]
        board_player_pov = np.roll(self.board,-self.current_player_ind,axis=0)
        player_move = self.get_player_move(player,board_player_pov)

        move_record = Transition(
            board_state = board_player_pov,
            selected_move = player_move,
            )
        self.game_data.append(move_record)

        return player_move

    
    def move_next_player_with(self,player_move):

        
        self.drop_in_slot(self.current_player_ind,player_move)
        player_won = self.check_win(self.current_player_ind)
        player_tie = self.board.sum() == self.board.shape[1]*self.board.shape[2]
        game_over = player_won or player_tie
        self.status = GameStatus.COMPLETE if game_over else self.status
        if len(self.game_data) >= 2 and not game_over:
            self.game_data[-2].resulting_state = self.board.copy()
        if player_won:
            self.game_data[-1].reward = 1
            self.winner = self.current_player_ind
        if player_tie:
            self.game_data[-1].reward = 0.5
            self.game_data[-2].reward = 0.5
        if self.status == GameStatus.INPROGRESS:
            self.move_ind += 1
            self.current_player_ind = self.move_ind % len(self.players)


    def next_player_make_move(self):
        if self.verbose:
            self.show_board()
        next_move = self.get_next_player_move()
        self.move_next_player_with(next_move)
    
    def play_game(self):
        
        self.start_game()
        while self.status == GameStatus.INPROGRESS:
            self.next_player_make_move()
            
        self.finish_game()
        return self.winner, self.game_data
                
    def finish_game(self):
        if self.verbose:
            self.show_board()
            if self.winner == -1:
                print("Game was a draw")
            else:
                print(f"Player {self.players[self.winner].name} won!")
            
        
