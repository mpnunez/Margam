from dataclasses import dataclass
import numpy as np

@dataclass
class MoveRecord:
    game_id: int = 0
    board_state: np.ndarray = np.zeros([2,6,7])
    legal_moves: list = None
    illegal_moves: list = None
    selected_move: int = 0
    move_ind: int = 0
    game_length: int = 0
    result: int = 0
    move_scores: np.ndarray() = np.zeros(7)
    
    def assign_scores(self,discount_rate=0.95):
        """
        Assign scores to each move based on game result
        """
        n_slots = self.board_state.shape[2]
        n_legal_moves = len(self.legal_moves)
        self.move_scores = np.zeros(n_slots)
        
        result_weight = discount_rate ** (self.game_length-self.move_ind)
        self.moves_scores[self.selected_move] = result_weight * self.result + (1-result_weight) * (1/n_legal_moves)
        
        remaining_score_balance = 1 - self.moves_scores.sum()
        unchosen_legal_moves = self.legal_moves.copy()
        unchosen_legal_moves.remove(self.selected_move)
        if unchosen_legal_moves:
            self.move_scores[unchosen_legal_moves] = remaining_score_balance / len(unchosen_legal_moves)
        
        return self.move_scores
    
    
if __name__ == "__main__":
    m = MoveRecord()
