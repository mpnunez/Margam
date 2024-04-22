from connect4lib.agents.player import Player
import numpy as np
import random

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.nn import softmax


class PolicyPlayer(Player):
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = None
    
    def initialize_model(self,n_rows,n_cols,n_players):
        input_shape = (n_rows,n_cols,n_players)
        nn_input = keras.Input(shape=input_shape)
        x = layers.Conv2D(64, 4)(nn_input)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        model_trunk_f = layers.Flatten()(x)
        
        x = layers.Dense(64, activation="relu")(model_trunk_f)
        logits_output = layers.Dense(n_cols, activation="linear")(x)
        
        x = layers.Dense(64, activation="relu")(model_trunk_f)
        state_value_output = layers.Dense(1, activation="linear")(x)
        
        self.model = keras.Model(inputs=nn_input, outputs=[logits_output,state_value_output], name="PGAC-model")
    
    def get_move(self,board: np.array, game) -> int:
        logits, state_value = self.model(board[np.newaxis,:])
        move_probabilities = softmax(logits[0])
        selected_move = random.choices(game.options, weights=move_probabilities, k=1)[0]
        return selected_move
