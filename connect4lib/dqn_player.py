from connect4lib.player import Player
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

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
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(loss="mse",
            optimizer= keras.optimizers.Adam(learning_rate=1e-2),
            metrics=["mse"])
        
    def train_on_game_data(self,move_records):
        x_train = np.stack([mr.board_state for mr in move_records])
        x_train = x_train.swapaxes(1,2).swapaxes(2,3)
        y_train = np.stack([mr.move_scores for mr in move_records])
        batch_size = 128
        epochs = 1

        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        
    def get_possible_next_positions(self,board) -> dict:
        """
        Return the boards that would result from 
        """
        next_positions = {}
        for col in range(7):
            for r in range(6):
                if np.sum(board[:,5-r,col])==0:
                    newboard = board.copy()
                    newboard[0,5-r,col] = 1
                    next_positions[col] = newboard
                    break

        return next_positions


    def get_move_scores_deterministic(self,board: np.array) -> np.array:
        next_positions = self.get_possible_next_positions(board)
        to_eval = np.array(list(next_positions.values()))
        q_values = self.model.predict(to_eval.swapaxes(1,2).swapaxes(2,3),verbose=0)
        move_scores = np.zeros(7)
        for col, q_value in zip(next_positions.keys(),q_values):
            move_scores[col] = q_value
        return move_scores
        
        
