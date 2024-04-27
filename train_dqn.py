from tqdm import tqdm
import numpy as np
from enum import Enum
import itertools
from collections import deque
from functools import partial

import tensorflow as tf
from tensorflow import one_hot
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorboardX import SummaryWriter
from keras.models import load_model

from connect4lib.game import Connect4, TicTacToe
from connect4lib.agents import RandomPlayer, ColumnSpammer
from connect4lib.agents import DQNPlayer
from connect4lib.agents.dqn_player import MiniMax
from connect4lib.hyperparams import *



def generate_transitions(agent, opponents):
    """
    Infinitely yield transitions by playing
    game episodes
    """

    for i, _ in enumerate(iter(bool, True)):

        opponent_ind = (i//2)%len(opponents)    # Play each opponent twice in a row
        opponent = opponents[opponent_ind]
        agent_position = i%2
        opponent_position = (agent_position+1)%2
        
        g = TicTacToe(nrows=NROWS,ncols=NCOLS,nconnectwins=NCONNECT)
        g.players = [None,None]
        g.players[agent_position] = agent        # Alternate being player 1/2
        g.players[opponent_position] = opponent   
        
        
        winner, records = g.play_game()
        agent_records = records[agent_position::len(g.players)]

        for move_record in agent_records:
            yield move_record, opponent





def sample_experience_buffer(buffer,batch_size):
    indices = np.random.choice(len(buffer), batch_size, replace=False)
    return [buffer[idx] for idx in indices]
        


def main():
    
    # Intialize players
    agent = DQNPlayer(name="Magnus")
    agent.initialize_model(NROWS,NCOLS,NPLAYERS)
    agent.model.summary()
    
    opponents = []

    experience_buffer = deque(maxlen=REPLAY_SIZE)
    reward_buffer = deque(maxlen=REWARD_BUFFER_SIZE)
    reward_buffer_vs = {}
    for opp in opponents:
        reward_buffer_vs[opp.name] = deque(maxlen=REWARD_BUFFER_SIZE//len(opponents))

    mse_loss=MeanSquaredError()
    optimizer= Adam(learning_rate=LEARNING_RATE)
    

    writer = SummaryWriter()
    best_reward = 0
    for frame_idx, (transition, opponent) in enumerate(generate_transitions(agent, opponents)):

        experience_buffer.append(transition)

        agent.random_weight = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        # Compute average reward
        if transition.resulting_state is None:
            reward_buffer.append(transition.reward)
            reward_buffer_vs[opponent.name].append(transition.reward)
            smoothed_reward = sum(reward_buffer) / len(reward_buffer)
            move_distribution = [mr.selected_move for mr in experience_buffer]
            move_distribution = np.array([move_distribution.count(i) for i in range(7)])
            move_distribution = move_distribution / move_distribution.sum()
            #print(f"Move distribution: {move_distribution}")
            writer.add_scalar("Average reward", smoothed_reward, frame_idx)
            for opp_name, opp_buffer in reward_buffer_vs.items():
                reward_vs = sum(opp_buffer) / len(opp_buffer) if len(opp_buffer) else 0
                writer.add_scalar(f"reward-vs-{opp_name}", reward_vs, frame_idx)
            if agent.random_weight > EPSILON_FINAL:
                writer.add_scalar("epsilon", agent.random_weight, frame_idx)

            if len(reward_buffer) == REWARD_BUFFER_SIZE and smoothed_reward > max(SAVE_MODEL_ABS_THRESHOLD,best_reward+SAVE_MODEL_REL_THRESHOLD):
                agent.model.save(f"magnus-{smoothed_reward}.keras")
                best_reward = smoothed_reward

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
        resulting_boards = np.array([mr.resulting_state if mr.resulting_state is not None else np.zeros(transition.board_state.shape) for mr in training_data])
        resulting_board_q = agent.target_network.predict_on_batch(resulting_boards.swapaxes(1,2).swapaxes(2,3))
        max_qs = np.max(resulting_board_q,axis=1)
        rewards = np.array([mr.reward for mr in training_data])
        q_to_train_single_values = rewards + DISCOUNT_RATE * np.multiply(non_terminal_states,max_qs)

        # Needed for our mask
        selected_moves = [mr.selected_move for mr in training_data]
        selected_move_mask = one_hot(selected_moves, NCOLS)
        
        # Compute MSE loss based on chosen move values only
        with tf.GradientTape() as tape:
            predicted_q_values = agent.model(x_train)
            predicted_q_values = tf.multiply(predicted_q_values,selected_move_mask)
            predicted_q_values = tf.reduce_sum(predicted_q_values, 1)
            q_prediction_errors = predicted_q_values - q_to_train_single_values
            loss_value = mse_loss(q_to_train_single_values, predicted_q_values)
 
        grads = tape.gradient(loss_value, agent.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, agent.model.trainable_variables))
        
        # Record prediction error
        writer.add_scalar("loss", loss_value.numpy(), frame_idx)
        if frame_idx % RECORD_HISTOGRAMS == 0:
            writer.add_histogram("q-error",q_prediction_errors.numpy(),frame_idx)

        # Update policy
        if frame_idx % SYNC_TARGET_NETWORK == 0:
            agent.target_network.set_weights(agent.model.get_weights())

    writer.close()

if __name__ == "__main__":
    main()
