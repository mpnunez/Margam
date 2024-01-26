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
from functools import partial

import tensorflow as tf
from tensorflow import one_hot
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

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
    
    NROWS = 6
    NCOLS = 7
    NCONNECT = 4

    agent = DQNPlayer(name="Magnus")
    #opponents = [RandomPlayer(name=f"RandomBot")]
    opponents = [ColumnSpammer(name=f"CS")]

    # DQN hyperparameters
    SAVE_MODEL_EVERY_N_TRANSITIONS = 100
    GAMMA = 0.99
    BATCH_SIZE = 32             
    REPLAY_SIZE = 1000
    LEARNING_RATE = 1e-3
    SYNC_TARGET_NETWORK = 100
    REPLAY_START_SIZE = 1000
    REWARD_BUFFER_SIZE = 100

    EPSILON_DECAY_LAST_FRAME = 10**4
    EPSILON_START = 1.0
    EPSILON_FINAL = 0.02

    # For debugging
    #REPLAY_START_SIZE = 20
    #BATCH_SIZE = REPLAY_START_SIZE

    experience_buffer = deque(maxlen=REPLAY_SIZE)
    reward_buffer = deque(maxlen=REWARD_BUFFER_SIZE)

    agent.model.compile(
        loss=MeanSquaredError(),
        optimizer= Adam(learning_rate=LEARNING_RATE))

    frame_idx = -1
    for transition in generate_transitions(agent, opponents):
        frame_idx += 1
        experience_buffer.append(transition)

        agent.random_weight = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        # Compute average reward
        if transition.resulting_state is None:
            reward_buffer.append(transition.reward)
            smoothed_reward = sum(reward_buffer) / len(reward_buffer)
            print(f"Average reward (last {len(reward_buffer)} games): {smoothed_reward}")
            move_distribution = [mr.selected_move for mr in experience_buffer]
            move_distribution = np.array([move_distribution.count(i) for i in range(7)])
            move_distribution = move_distribution / move_distribution.sum()
            print(f"Epsilon: {agent.random_weight}")
            print(f"Move distribution: {move_distribution}")

        # Don't start training the network until we have enough data
        if len(experience_buffer) < REPLAY_START_SIZE:
            continue

        training_data = sample_experience_buffer(experience_buffer,BATCH_SIZE)

        # Make X and Y
        x_train = np.array([mr.board_state for mr in training_data])
        x_train = x_train.swapaxes(1,2).swapaxes(2,3)

        # Bellman equation part
        # Take maximum Q(s',a') of board states we end up in
        non_terminal_states = np.array([mr.resulting_state is not None for mr in training_data])
        resulting_boards = np.array([mr.resulting_state if mr.resulting_state is not None else np.zeros([2,6,7]) for mr in training_data])
        resulting_board_q = agent.target_network.predict(resulting_boards.swapaxes(1,2).swapaxes(2,3),verbose=0)
        max_qs = np.max(resulting_board_q,axis=1)
        rewards = np.array([mr.reward for mr in training_data])
        q_to_train = rewards + GAMMA * np.multiply(non_terminal_states,max_qs)

        # Needed for our mask
        selected_moves = [mr.selected_move for mr in training_data]
        selected_move_mask = one_hot(selected_moves, NCOLS)
        q_to_train_mat = q_to_train[:,np.newaxis]*selected_move_mask

        # Hack to mask q-values for unselected moves
        # Set training value equal to predicted value
        # Loss and gradient will be zero for MSE
        unselected_move_mask = np.ones(selected_move_mask.shape) - selected_move_mask
        q_predicted = agent.model.predict_on_batch(x_train)
        q_to_train_mat = q_to_train_mat*selected_move_mask + q_predicted*unselected_move_mask

        # Step the gradients
        agent.model.train_on_batch(x_train,q_to_train_mat)

        # Update policy
        if frame_idx % SYNC_TARGET_NETWORK == 0:
            agent.target_network.set_weights(agent.model.get_weights())

if __name__ == "__main__":
    main()

