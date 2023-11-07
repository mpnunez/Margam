from player import Player
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

class AIPlayer(Player):
    
    def __init__self(name=None):
        super().__init__(name)
        self.model = None
    
    

    def train_on_game_data(self,move_records):
        return
        self.model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    
    def get_move_scores(self,board: np.array) -> np.array:
        n_cols = board.shape[2]
        random_scores = np.random.rand(n_cols)
        random_scores = random_scores / random_scores.sum()
        return random_scores
        
    def get_move_scores_ai(self,board: np.array) -> np.array:
        
        # pre-process board
        board_postproc = board.copy()
        move_scores = self.model.predict(board_postproc)
        
        return move_scores
        
        
