from player import Player
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

class AIPlayer(Player):
    
    def __init__(self,name=None):
        super().__init__(name)
        
        num_classes = 7
        input_shape = (6, 7, 2)
        self.model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Flatten(),
                layers.Dense(50, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
    
    
    def transform_board_state(self,baord_state: np.ndarray):
        """
        Transform board to dimensions needed by CNN
        """
        pass
        

    def train_on_game_data(self,move_records):
        x_train = np.stack([mr.board_state for mr in move_records])
        x_train = x_train.swapaxes(1,2).swapaxes(2,3)
        
        y_train = np.stack([mr.move_scores for mr in move_records])
        
        
        
        
        
        batch_size = 128
        epochs = 15
        
        self.model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_absolute_error"])
        
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        
    def get_move_scores(self,board: np.array) -> np.array:

        # pre-process board
        board_postproc = np.array([board.copy()])
        board_postproc = board_postproc.swapaxes(1,2).swapaxes(2,3)
        move_scores = self.model.predict(board_postproc,verbose=0)[0]
        
        
        return move_scores
        
        
