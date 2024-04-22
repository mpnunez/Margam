from connect4lib.gents.player import Player
import numpy as np
import random

from tensorflow import keras
from tensorflow.keras import layers

import copy

class DQNPlayer(Player):
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = None
        self.target_network = None
        self.random_weight = 0
    
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

    
    def get_move(self,board: np.array, game) -> int:
        q_values = self.model.predict_on_batch(board)[0]
        max_q_ind = np.argmax(q_values)
        if (self.random_weight != 0) and (np.random.random() < self.random_weight):
            return random.choice(game.options)
        return options[max_q_ind]
