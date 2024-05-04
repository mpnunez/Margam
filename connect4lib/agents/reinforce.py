from connect4lib.agents.player import Player
import numpy as np
import random

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.nn import softmax


class ReinforcePlayer(Player):
    # for tic tac toe
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = None
    
    def initialize_model(self,n_rows,n_cols,n_players,n_outputs):
        input_shape = (n_rows,n_cols,n_players)
        self.model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dense(n_outputs, activation="linear"),
            ]
        )

    def get_move(self,board: np.array, game) -> int:
        logits = self.model(board[np.newaxis,:])[0]
        move_probabilities = softmax(logits)
        selected_move = random.choices(game.options, weights=move_probabilities, k=1)[0]
        return selected_move
