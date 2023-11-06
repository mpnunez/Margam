from player import Player
import numpy as np

class AIPlayer(Player):
    def get_move_scores(self,board: np.array) -> np.array:
        n_cols = board.shape[2]
        random_scores = np.random.rand(n_cols)
        random_scores = random_scores / random_scores.sum()
        return random_scores

    def train_on_game_data(self,move_records):
        for mr in move_records:
            pass
