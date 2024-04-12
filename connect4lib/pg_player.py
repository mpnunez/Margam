from connect4lib.player import Player
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

import copy

class PolicyPlayer(Player):
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = None
        self.target_network = None
    
    def initialize_model(self,n_rows,n_cols,n_players):
        input_shape = (n_rows,n_cols,n_players)
        self.model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(64, 4),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dense(n_cols, activation="linear"),
            ]
        )
        self.target_network = keras.models.clone_model(self.model)
        self.target_network.set_weights(self.model.get_weights())

    
    def get_move_scores_deterministic(self,board: np.array) -> np.array:
        """
        TODO: rethink how the game handles illegal moves
        requested by 
        Is ranking the best way? 
        """
        move_scores = self.model.predict_on_batch(board.swapaxes(0,1).swapaxes(1,2)[np.newaxis,:])[0]
        return move_scores
