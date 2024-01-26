from connect4lib.player import Player
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

import copy

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
        move_scores = self.model.predict_on_batch(board.swapaxes(0,1).swapaxes(1,2)[np.newaxis,:])[0]
        return move_scores
