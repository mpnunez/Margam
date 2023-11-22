from game import Game
from player import RandomPlayer
from aiplayer import AIPlayer
from tqdm import tqdm
import pickle
import numpy as np

def play_matches(trainee, opponents, n_games=100):
    all_move_records = []
    trainee_wlt_record = np.zeros([len(opponents),3])
    WIN = 0
    LOSS = 1
    TIE = 2
    
    for i in tqdm(range(n_games)):
        #print(f"{i}/{n_training_games}")
        
        opponent_ind = (i//2)%len(opponents)    # Play each opponent twice in a row
        opponent = opponents[opponent_ind]
        trainee_position = i%2
        opponent_position = (trainee_position+1)%2
        
        g = Game()
        g.players = [None,None]
        g.players[trainee_position] = trainee        # Alternate being player 1/2
        g.players[opponent_position] = opponent   
        
        
        winner, records = g.play_game()
        
        for mr in records:
            mr.assign_scores()
        all_move_records += records
        
        if winner == -1:
            trainee_wlt_record[opponent_ind,TIE] += 1
        elif winner == trainee_position:
            trainee_wlt_record[opponent_ind,WIN] += 1
        else:
            trainee_wlt_record[opponent_ind,LOSS] += 1

        
    return all_move_records, trainee_wlt_record
        

def main():
    
    magnus = AIPlayer(name="Magnus")
    random_bot = RandomPlayer("Random Bot")
    opponents = [random_bot]
    
    n_training_rounds = 3
    for training_round in range(n_training_rounds):
    
        trainee = random_bot if training_round == 0 else magnus    # Use random bot in 1st round to save time 
        all_move_records, win_loss_ties = play_matches(trainee, opponents, n_games=100)
        print("Opponents:")
        print([op.name for op in opponents])
        print(win_loss_ties)
            
        magnus.train_on_game_data(all_move_records)
        chkpt_fname = f'magnus-weights-{training_round}.ckpt'
        magnus.model.save_weights(chkpt_fname)
        
        # Add copy of self to opponent list
        magnus_clone = AIPlayer(name=f"Magnus-{training_round}")
        magnus_clone.model.load_weights(chkpt_fname).expect_partial()
        opponents.append(magnus_clone)

if __name__ == "__main__":
    main()

