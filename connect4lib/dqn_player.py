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
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                #layers.Dense(64, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(7, activation="linear",bias_initializer=keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=None)),
            ]
        )
        self.target_network = keras.models.clone_model(self.model)
        self.target_network.set_weights(self.model.get_weights())
        print(self.model.summary())

    
    def get_move_scores_deterministic(self,board: np.array) -> np.array:
        move_scores = self.model.predict_on_batch(board.swapaxes(0,1).swapaxes(1,2)[np.newaxis,:])[0]
        return move_scores
