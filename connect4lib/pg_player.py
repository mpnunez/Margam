from connect4lib.player import Player
import numpy as np
import random

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.nn import softmax

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
    
    def get_move(self,board: np.array) -> np.array:
        n_cols = board.shape[2]
        logits = self.model.predict_on_batch(board.swapaxes(0,1).swapaxes(1,2)[np.newaxis,:])[0]
        move_probabilities = softmax(logits)
        selected_move = random.choices(range(n_cols), weights=move_probabilities, k=1)[0]
        return selected_move
