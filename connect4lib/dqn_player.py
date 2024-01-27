from connect4lib.player import Player
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

import copy

def encode_position_action(board,col):
    """
    If move is legal, return the resulting board. Return None
    for illegal moves.
    """
    n_rows = board.shape[1]
    for r in range(n_rows):
        row_to_try = n_rows-r-1
        if np.sum(board[:,row_to_try,col])>0:
            continue

        newboard = board.copy()
        newboard[0,row_to_try,col] = 1
        return newboard

class DQNPlayer(Player):
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        input_shape = (6, 7, 2)
        self.model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(128, 4),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(7, activation="linear"),
            ]
        )
        self.model.compile(loss="mse",
            optimizer= keras.optimizers.Adam(learning_rate=1e-2),
            metrics=["mse"])
        #self.target_network = copy.deepcopy(self.model) # for training

    
    def get_move_scores_deterministic(self,board: np.array) -> np.array:
        x = np.array([board])
        move_scores = self.model.predict(x.swapaxes(1,2).swapaxes(2,3),verbose=0)[0]
        return move_scores
