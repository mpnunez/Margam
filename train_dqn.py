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
    indices = np.random.choice(len(buffer), batch_size, replace=False)
    return [buffer[idx] for idx in indices]
        

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

    # For debugging
    REPLAY_START_SIZE = 5
    BATCH_SIZE = REPLAY_START_SIZE

    experience_buffer = deque(maxlen=REPLAY_SIZE)
    for transition in generate_transitions(agent, opponents):
        experience_buffer.append(transition)

        if len(experience_buffer) < REPLAY_START_SIZE:
            continue

        training_data = sample_experience_buffer(experience_buffer,BATCH_SIZE)
        print(training_data[0])

        # Make X and Y
        X = np.array([mr.board_state for mr in training_data])

        # Bellman equation part
        # Take maximum Q(s',a') of board states we end up in
        resulting_boards = np.array([mr.resulting_state for mr in training_data])
        print(resulting_boards.shape)
        resulting_board_q = agent.model.predict(resulting_boards.swapaxes(1,2).swapaxes(2,3),verbose=0)
        print(resulting_board_q)
        max_qs = np.max(resulting_board_q,axis=1)
        print(max_qs)

        rewards = np.array([mr.reward for mr in training_data])
        q_to_train = rewards + GAMMA * max_qs
        print(q_to_train)

        # Needed for our mask
        y = [mr.selected_move for mr in training_data]
        #print(y)

        break


if __name__ == "__main__":
    main()

