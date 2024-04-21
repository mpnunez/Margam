import numpy as np
from connect4lib.utils import Connect4Exception
from connect4lib.transition import Transition
from enum import Enum
import random

from abc import ABC, abstractmethod

class GameStatus(Enum):
    NOTSTARTED = 1
    INPROGRESS = 2
    COMPLETE = 3

class Game(ABC):
    def __init__(self,nrows=6,ncols=7,nconnectwins=4):
        self.nrows = nrows
        self.ncols = ncols
        self.nconnectwins = nconnectwins
        self.board = None
        self.current_player_ind = 0
        self.players = []
        self.status = GameStatus.NOTSTARTED
        self.game_data = []
        self.winner = -1
        self.move_ind = 0
        self.verbose = False
        self.WIN_REWARD = 1
        self.TIE_REWARD = 0
        self.LOSS_REWARD = -1

    def _check_horizontal_win(self,player: int) -> bool:
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
    
    def _check_vertical_win(self,player: int) -> bool:
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
    
    def _check_diagonal_ll_ur_win(self,player: int) -> bool:
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
    
    def _check_diagonal_ul_lr_win(self,player: int) -> bool:
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


    def check_win(self,player: int) -> bool:
        return self._check_horizontal_win(player) or self._check_vertical_win(player) or self._check_diagonal_ll_ur_win(player) or self._check_diagonal_ul_lr_win(player)
    
    @abstractmethod
    def drop_in_slot(self,player: int, col: int):
        pass
    
    def show_board(self):
        display_board = sum((i+1)*self.board[:,:,i] for i in range(self.board.shape[2])))
        print(display_board)
        print()

    @abstractmethod   
    def get_legal_illegal_moves(self):
        pass

    @abstractmethod
    def get_player_move(self,player,board_player_pov):
        pass

        
    
    def start_game(self):
        self.board = np.zeros([self.nrows,self.ncols,len(self.players)])
        self.status = GameStatus.INPROGRESS
    
    def get_board_from_player_pov(self, player_ind: int) -> np.array:
        return np.roll(self.board,-player_ind,axis=2)

    def get_next_player_move(self):
        player = self.players[self.current_player_ind]
        board_player_pov = self.get_board_from_player_pov(self.current_player_ind)
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
        player_tie = self.board.sum() == self.board.shape[0]*self.board.shape[1]
        game_over = player_won or player_tie
        self.status = GameStatus.COMPLETE if game_over else self.status
        if len(self.game_data) >= 2 and not game_over:
            self.game_data[-2].resulting_state = self.get_board_from_player_pov((self.current_player_ind-1)%len(self.players))
        if player_won:
            self.winner = self.current_player_ind
            self.game_data[-1].reward = self.WIN_REWARD
            for i in range(max(-len(self.players),-len(self.game_data)),-1):
                self.game_data[i].reward = self.LOSS_REWARD
        if player_tie:
            for i in range(max(-len(self.players),-len(self.game_data)),0):
                self.game_data[i].reward = self.TIE_REWARD
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
        if not self.verbose:
            return
            
        self.show_board()
        if self.winner == -1:
            print("Game was a draw")
        else:
            print(f"Player {self.players[self.winner].name} won!")
