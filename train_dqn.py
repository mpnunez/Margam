from connect4lib.game import Game
from connect4lib.player import RandomPlayer, ColumnSpammer
from connect4lib.dqn_player import DQNPlayer
from tqdm import tqdm
import numpy as np
from keras.models import load_model
from multiprocessing import Pool

def play_match(trainee,opponents,i):
    opponent_ind = (i//2)%len(opponents)    # Play each opponent twice in a row
    opponent = opponents[opponent_ind]
    trainee_position = i%2
    opponent_position = (trainee_position+1)%2
    
    g = Game()
    g.players = [None,None]
    g.players[trainee_position] = trainee        # Alternate being player 1/2
    g.players[opponent_position] = opponent   
    
    
    winner, records = g.play_game()

    return winner, trainee_position, opponent_ind, records

def play_matches(trainee, opponents, n_games=100):
    all_move_records = []
    trainee_wlt_record = np.zeros([len(opponents),3],dtype=int)
    WIN = 0
    LOSS = 1
    TIE = 2

    print(f"Playing {n_games} games...")
    use_multiprocessing = False
    
    if use_multiprocessing:
        with Pool(5) as p:
            game_results = p.map(play_match, [(trainee,opponents,i) for i in range(n_games)])
    else:
        game_results = [play_match(trainee,opponents,i) for i in tqdm(range(n_games))]

    # Accumulate results
    for winner, trainee_position, opponent_ind, records in game_results:
        all_move_records += records
        
        if winner == -1:
            trainee_wlt_record[opponent_ind,TIE] += 1
        elif winner == trainee_position:
            trainee_wlt_record[opponent_ind,WIN] += 1
        else:
            trainee_wlt_record[opponent_ind,LOSS] += 1

        
    return all_move_records, trainee_wlt_record
        

def main():
    
    magnus = DQNPlayer(name="Magnus")
    #magnus.random_weight = 0.2
    random_bot = RandomPlayer("Random Bot")
    #opponents = [random_bot, ColumnSpammer(name="ColumnSpammer")]
    #opponents = [ColumnSpammer(name=f"ColumnSpammer-{i}",col_preference=i) for i in range(7)]
    #opponents += [RandomPlayer(f"RandomBot_{i}") for i in range(7)]
    #opponents = [ColumnSpammer(name=f"ColumnSpammer",col_preference=4)]
    opponents = [RandomPlayer(name=f"Random Bot")]
    self_play = False
    percentile_keep = 0.3       # Train on this fraction of best games
    SAVE_MODEL_EVERY_N_BATCHES = 100
    #GAMES_PER_TRAINING_BATCH = 100
    GAMES_PER_TRAINING_BATCH = 10
    N_TRAINING_BATCHES = 1000

    for training_round in range(N_TRAINING_BATCHES):
    
        all_move_records, win_loss_ties = play_matches(magnus, opponents, n_games=GAMES_PER_TRAINING_BATCH)

        # Print table of win/loss/tie/records
        print("Opponent\tWins\tLosses\tTies")
        for op, row in zip(opponents,win_loss_ties):
            print(f"{op.name}\t",end='')
            for col in row:
                print(f"{col}\t",end='')
            print()
            
        agent_move_records = [mr for mr in all_move_records if mr.player_name == "Magnus"]
        y_train = np.stack([mr.move_scores for mr in agent_move_records])
        print(np.sum(y_train,axis=0))

        return

if __name__ == "__main__":
    main()

