import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap

from player import RandomPlayer, HumanPlayer
from game import Game
from player import Player
import numpy as np
import time


class HumanGUIPlayer(Player):
    
    def __init__(self,name=None,gui=None):
        super().__init__(name)
        self.gui = gui
    
    def get_move_scores(self,board: np.array) -> np.array:
        n_cols = board.shape[2]
        invalid_input = True
        while invalid_input:
            new_input = input(f"Select column to drop token into [0-{n_cols-1}]\n")
            try:
                slot_to_drop = int(new_input)
            except ValueError:
                continue
            
            if 0 <= slot_to_drop and slot_to_drop < n_cols:
                invalid_input = False
                
        scores = np.zeros(n_cols)
        scores[slot_to_drop] = 1
        return scores
    
    def get_move_scores_gui(self,board: np.array) -> np.array:
        n_cols = board.shape[2]
        
        # Waiting for button press
        slot_to_drop = None
        while slot_to_drop is None:
            #slot_to_drop = gui.last_button_pressed() # ??
            return np.array([1,0,0,0,0,0,0])
            time.sleep(0.10)
                
        scores = np.zeros(n_cols)
        scores[slot_to_drop] = 1
        return scores

def window(game: Game, on_click):
    app = QApplication(sys.argv)
    win = QWidget()
    grid = QGridLayout()
    	
    nrows = 6
    ncols = 7
    for i in range(nrows):
       for j in range(ncols):
           pixmap = QPixmap('empty.png')
           pixmap = pixmap.scaledToWidth(100)
           label = QLabel()
           label.setPixmap(pixmap)
           grid.addWidget(label,i,j)
           
    for j in range(ncols):
        grid.addWidget(QPushButton("Drop"),nrows,j)
        
    # Start game button
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
    human = HumanGUIPlayer()
    g.players = [human,RandomPlayer()]
    
    @pyqtSlot()
    def on_click(self):
        g.play_game(show_board_each_move=True,verbose=True)
    
    window(g,on_click)
    
    

if __name__ == '__main__':
   main()