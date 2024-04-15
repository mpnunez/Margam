from connect4lib.player import Player
import numpy as np
import random

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.nn import softmax

from typing import Tuple

import copy

class PolicyPlayer(Player):
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = None
        self.target_network = None
    
    def initialize_model(self,n_rows,n_cols,n_players):
        input_shape = (n_rows,n_cols,n_players)
        nn_input = keras.Input(shape=input_shape)
        x = layers.Conv2D(64, 4)(nn_input)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        self.model_trunk_f = layers.Flatten()(x)
        self.model_trunk = keras.Model(inputs=nn_input, outputs=self.model_trunk_f, name="trunk")

        x = layers.Dense(64, activation="relu")(self.model_trunk_f)
        self.logits_model_f = layers.Dense(n_cols, activation="linear")(x)
        self.logits_model = keras.Model(inputs=self.model_trunk_f, outputs=self.logits_model_f, name="logits")

        x = layers.Dense(64, activation="relu")(self.model_trunk_f)
        self.state_value_model_f = layers.Dense(1, activation="linear")(x)
        self.state_value_model = keras.Model(inputs=self.model_trunk_f, outputs=self.state_value_model_f, name="state value")

    def predict(self,board:np.array) -> Tuple[np.array,float]:

        trunk_value = self.model_trunk(board.swapaxes(0,1).swapaxes(1,2)[np.newaxis,:])
        logits = self.logits_model(trunk_value)[0]
        state_value = self.state_value_model(trunk_value)[0]

        return logits, state_value
    
    def get_move(self,board: np.array) -> np.array:
        n_cols = board.shape[2]
        logits, state_value = self.predict(board)
        move_probabilities = softmax(logits)
        selected_move = random.choices(range(n_cols), weights=move_probabilities, k=1)[0]
        return selected_move
