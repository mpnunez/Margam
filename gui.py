import sys
import time

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QPushButton, QComboBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, QThread, QObject, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMessageBox

from connect4lib.player import RandomPlayer, HumanPlayer
from connect4lib.game import Game, GameStatus
from connect4lib.player import Player, ColumnSpammer, RandomPlayer
from connect4lib.dqn_player import DQNPlayer
import numpy as np
from keras.models import load_model


import time

import functools

"""

When the game starts or the user presses a button,
loop through next players until the game ends or until you 
reach a human player

"""

class HumanGUIPlayer(Player):
    requires_user_input = True
    def __init__(self,name=None,gui=None):
        super().__init__(name)
        self.next_move = 0
    
    def get_move(self,board: np.array) -> np.array:
        return self.next_move

class Connect4GUI(QWidget):
    def __init__(self,parent=None):
        QWidget.__init__(self, parent)
        
        #self.thread = Worker()
        
        
        self.game = Game()
        human = HumanGUIPlayer(name="Human")
        magnus = DQNPlayer(name="Magnus")
        magnus.model = load_model('magnus-0.832.keras')
        self.game.players = [human,magnus]
        #self.game.verbose = True


        
        nrows = 6
        ncols = 7
        #_, nrows, ncols = self.game.board.shape
        self.nrows = nrows
        self.ncols = ncols
        self.grid = QGridLayout()
        
        
        
        self.pix_maps = {}
        for color in ("empty","red","blue"):
            self.pix_maps[color] = QPixmap(f'assets/{color}.png')
            self.pix_maps[color] = self.pix_maps[color].scaledToWidth(100)
        
        self.label_grid = []
        for i in range(nrows):
            label_row = []
            for j in range(ncols):
            
                label = QLabel()
                label.setPixmap(self.pix_maps["empty"])
                self.grid.addWidget(label,i,j)
                label_row.append(label)
            
            self.label_grid.append(label_row)
    
    
        
    
        self.msg = QMessageBox()
        self.msg.setWindowTitle("Connect4")
        
    
        
            
        # Add drop buttons
        for j in range(ncols):
            drop_button = QPushButton("Drop")
            self.grid.addWidget(drop_button,nrows,j)
            drop_button.clicked.connect(functools.partial(self.make_human_move, j))
            drop_button.clicked.connect(self.move_until_next_human_player)
            
        
        # Add New Game button
        start_button = QPushButton("New Game")
        start_button.clicked.connect(self.start_new_game)
        self.grid.addWidget(start_button,nrows+1,0)
    

        def set_player(player_text,ind):

            if player_text == 'Human':
                self.game.players[ind] = HumanGUIPlayer()
            elif player_text == 'Random':
                self.game.players[ind] = RandomPlayer()
            elif player_text == 'Column Spammer':
                self.game.players[ind] = ColumnSpammer()
            elif player_text == 'Load model':
                self.game.players[ind] = DQNPlayer()
                self.game.players[ind].model = load_model("magnus.keras")

            


        # Add player selectors
        player_selectors = [QComboBox(), QComboBox()]
        for ind, ps in enumerate(player_selectors):
            ps.addItems(['Human', 'Random', 'Column Spammer', 'Load model'])
            self.grid.addWidget(ps,nrows+1,ind+1)
            ps.setEditable(True) 
            ps.currentTextChanged.connect(functools.partial(set_player, ind=ind))
        			
        
        # Initialize layout
        self.setLayout(self.grid)
        self.setWindowTitle("PyQt Grid Example")
        self.setGeometry(50,50,200,200)
    
    def update_board(self):
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.game.board[0,i,j] == 1:
                    pixmap = self.pix_maps["red"]
                elif self.game.board[1,i,j] == 1:
                    pixmap = self.pix_maps["blue"]
                else:
                    pixmap = self.pix_maps["empty"]
                pixmap = pixmap.scaledToWidth(100)
                self.label_grid[i][j].setPixmap(pixmap)
                

    def reportProgress(self,move_ind):
        self.game.move_next_player_with(move_ind)
        self.update_board()

    def start_new_game(self):
        self.game.start_game()
        self.move_until_next_human_player()
        self.update_board()
        
    def check_game_completion(self) -> bool:
        if self.game.status == GameStatus.COMPLETE:
            self.game.finish_game()
            self.msg.setText("Game is complete!")
            self.msg.exec_()
            
        return self.game.status == GameStatus.COMPLETE

    

    def make_human_move(self,j):
        if self.game.status == GameStatus.NOTSTARTED:
            self.msg.setText("Press 'New Game' to start game")
            self.msg.exec_()
            return
        if self.check_game_completion():
            return
        
        if not self.game.players[self.game.current_player_ind].requires_user_input:
            return
        
        self.game.players[self.game.current_player_ind].next_move = j
        self.game.next_player_make_move()
        self.update_board()
        self.check_game_completion()
        

    def move_until_next_human_player(self):
        
            
        # Step 2: Create a QThread object
        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = Worker(self)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.reportProgress)
        # Step 6: Start the thread
        self.thread.start()


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self,gui):
        super().__init__()
        self.gui = gui

    def run(self):
        """Long-running task."""
        if self.gui.game.status == GameStatus.INPROGRESS and not self.gui.game.players[self.gui.game.current_player_ind].requires_user_input:
            time.sleep(1)
            move_ind = self.gui.game.get_next_player_move()
            self.progress.emit(move_ind)


        self.finished.emit()
   
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Connect4GUI()
    window.show()
    sys.exit(app.exec_())