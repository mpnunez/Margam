import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap

from aiplayer import AIPlayer
from player import RandomPlayer, HumanPlayer
from game import Game, GameStatus
from player import Player
import numpy as np
import time

import functools
import pickle

class HumanGUIPlayer(Player):
    
    def __init__(self,name=None,gui=None):
        super().__init__(name)
        self.next_move = 0
    
    def get_move_scores(self,board: np.array) -> np.array:
        n_cols = board.shape[2]
        move_scores = np.zeros(n_cols)
        move_scores[self.next_move] = 1
        return move_scores

def window(game: Game):
    app = QApplication(sys.argv)
    win = QWidget()
    grid = QGridLayout()
    	
    nrows = 6
    ncols = 7
    
    
    empty_pixmap = QPixmap('empty.png')
    empty_pixmap = empty_pixmap.scaledToWidth(100)
    red_pixmap = QPixmap('empty.png')
    red_pixmap = red_pixmap.scaledToWidth(100)
    blue_pixmap = QPixmap('empty.png')
    blue_pixmap = blue_pixmap.scaledToWidth(100)
    
    label_grid = []
    
    for i in range(nrows):
        label_row = []
        for j in range(ncols):
           
            label = QLabel()
            label.setPixmap(empty_pixmap)
            grid.addWidget(label,i,j)
            label_row.append(label)
           
        label_grid.append(label_row)
    
    def update_board():
        for i in range(nrows):
            for j in range(ncols):
                if game.board[0,i,j] == 1:
                    pixmap = QPixmap('red.png')
                elif game.board[1,i,j] == 1:
                    pixmap = QPixmap('blue.png')
                else:
                    pixmap = QPixmap('empty.png')
                pixmap = pixmap.scaledToWidth(100)
                label_grid[i][j].setPixmap(pixmap)
    
    @pyqtSlot()
    def change_picture(j):
        if game.status == GameStatus.COMPLETE:
            print("Game is done!")
            return
        game.players[0].next_move = j
        
        game.next_player_make_move()
        update_board()    # Does not update in GUI event loop until this function is complete
        if game.status == GameStatus.COMPLETE:
            game.finish_game()
            return
        #time.sleep(1)
        game.next_player_make_move()
        update_board()
        if game.status == GameStatus.COMPLETE:
            game.finish_game()
            return
        
           
    for j in range(ncols):
        drop_button = QPushButton("Drop")
        grid.addWidget(drop_button,nrows,j)
        drop_button.clicked.connect(functools.partial(change_picture, j))
        
    # Start game button
    @pyqtSlot()
    def on_click():
        game.start_game()

    start_button = QPushButton("Start Game")
    start_button.clicked.connect(on_click)
    grid.addWidget(start_button,nrows+1,0)
        
    			
    win.setLayout(grid)
    win.setWindowTitle("PyQt Grid Example")
    win.setGeometry(50,50,200,200)
    win.show()
    
    #game.play_game(show_board_each_move=True)
    
    sys.exit(app.exec_())

def main():
    g = Game()
    human = HumanGUIPlayer(name="Human")
    magnus = AIPlayer(name="Magnus")
    magnus.model.load_weights('magnus_weights.ckpt')
    g.players = [human,magnus]
    g.verbose = True

    window(g)
    
    

if __name__ == '__main__':
   main()