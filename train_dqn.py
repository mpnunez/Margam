from connect4lib.game import Game
from connect4lib.player import RandomPlayer, ColumnSpammer
from connect4lib.dqn_player import DQNPlayer
from tqdm import tqdm
import numpy as np
from keras.models import load_model
from collections import Counter
from enum import Enum
import itertools
from collections import deque

def play_match(agent,opponents,i):
    """
    Play a single game and return
    transition records for the agent
    """
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

    return agent_records

def generate_transitions(agent, opponents):
    """
    Infinitely yield transitions by playing
    game episodes
    """
    i = 0
    while True:
        agent_records = play_match(agent,opponents,i)
        for move_record in agent_records:
            yield move_record
        i += 1

    return agent_move_records, agent_wlt_record

def sample_experience_buffer(buffer,batch_size):
    indices = np.random.choice(len(self.buffer), batch_size, replace=False)
    return [buffer[idx] for idx in indices])
        

def main():
    
    agent = DQNPlayer(name="Magnus")
    opponents = [RandomPlayer(name=f"RandomBot")]

    # DQN hyperparameters
    SAVE_MODEL_EVERY_N_TRANSITIONS = 100
    GAMMA = 0.99
    BATCH_SIZE = 32             
    REPLAY_SIZE = 10000
    LEARNING_RATE = 1e-4
    SYNC_TARGET_NETWORK = 1000
    REPLAY_START_SIZE = 10000

    experience_buffer = deque(maxlen=REPLAY_SIZE)
    for transition in generate_transitions(agent, opponents):
        experience_buffer.append(transition)

        if len(experience_buffer) < REPLAY_START_SIZE:
            continue

        training_data = sample_experience_buffer(experience_buffer,BATCH_SIZE)
        break


if __name__ == "__main__":
    main()

