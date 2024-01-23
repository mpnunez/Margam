from connect4lib.player import Player
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

class AIPlayer(Player):
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        num_classes = 7
        input_shape = (6, 7, 2)
        self.model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(128, 4),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
        self.model.compile(loss="categorical_crossentropy",
            optimizer= keras.optimizers.Adam(learning_rate=1e-2),
            metrics=["categorical_accuracy"])
        
    def train_on_game_data(self,move_records):
        x_train = np.stack([mr.board_state for mr in move_records])
        x_train = x_train.swapaxes(1,2).swapaxes(2,3)
        y_train = np.stack([mr.move_scores for mr in move_records])
        batch_size = 128
        epochs = 1

        
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        
    def get_move_scores_deterministic(self,board: np.array) -> np.array:

        move_scores = self.model.predict(np.array([board.swapaxes(0,1).swapaxes(1,2)]),verbose=0)[0]
        
        return move_scores
        
        
