from connect4lib.game import Game
from connect4lib.player import RandomPlayer, ColumnSpammer
from connect4lib.dqn_player import DQNPlayer
from tqdm import tqdm
import numpy as np
from keras.models import load_model
from collections import Counter
from enum import Enum
import itertools

def play_match(agent,opponents,i):
    opponent_ind = (i//2)%len(opponents)    # Play each opponent twice in a row
    opponent = opponents[opponent_ind]
    agent_position = i%2
    opponent_position = (agent_position+1)%2
    
    g = Game()
    g.players = [None,None]
    g.players[agent_position] = agent        # Alternate being player 1/2
    g.players[opponent_position] = opponent   
    
    
    winner, records = g.play_game()
    agent_records = records[agent_position::len(g.players)]

    return agent_records, agent_records[-1].reward, opponent_ind

def play_matches(agent, opponents, n_games=100):
    agent_move_records = []
    agent_wlt_record = np.zeros([len(opponents),3],dtype=int)
    
    print(f"Playing {n_games} games...")
    game_results = (play_match(agent,opponents,i) for i in tqdm(range(n_games)))

    for agent_records, agent_reward, opponent_ind in game_results:
        agent_move_records += agent_records
        reward_to_table_ind = {1: 0, 0: 1, 0.5: 2}  # win / loss / tie
        agent_wlt_record[opponent_ind,reward_to_table_ind[agent_reward]] += 1

    return agent_move_records, agent_wlt_record
        

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
    
        agent_move_records, win_loss_ties = play_matches(magnus, opponents, n_games=GAMES_PER_TRAINING_BATCH)

        # Print table of win/loss/tie/records
        print("Opponent\tWins\tLosses\tTies")
        for op, row in zip(opponents,win_loss_ties):
            print(f"{op.name}\t",end='')
            for col in row:
                print(f"{col}\t",end='')
            print()
            
        # View distribution of agent moves
        selected_moves = [mr.selected_move for mr in agent_move_records]
        move_counts = [selected_moves.count(i) for i in range(7)]
        print(move_counts)


        return

if __name__ == "__main__":
    main()

