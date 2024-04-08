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

from connect4lib.game import Game
from connect4lib.player import RandomPlayer, ColumnSpammer
from connect4lib.dqn_player import DQNPlayer
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
        
        g = Game(nrows=NROWS,ncols=NCOLS,nconnectwins=NCONNECT)
        g.players = [None,None]
        g.players[agent_position] = agent        # Alternate being player 1/2
        g.players[opponent_position] = opponent   
        
        
        winner, records = g.play_game()
        agent_records = records[agent_position::len(g.players)]

        for move_record in agent_records:
            yield move_record, opponent


def generate_transitions_pg(agent, opponents):
    """
    Sample a full episode, then assign q values
    based on final reward
    """

    episode_transitions = []
    q_values = []
    for transition, opponent in generate_transitions(agent,opponents):

        episode_transitions.append(transition)

        if transition.resulting_state is not None:
            continue

        # Assign q-values based on unrolling to final state
        prev_q = 0
        for tsn in reversed(episode_transitions):
            q = tsn.reward + DISCOUNT_RATE * prev_q
            prev_q = q
            q_values.append(q)
        q_values = list(reversed(q_values))
            
        # Yield scored transitions to caller
        for tsn, q_value in zip(episode_transitions,q_values):
            yield tsn, q_value, opponent

        # Reset for next episode
        episode_transitions = []
        q_values = []


def sample_experience_buffer(buffer,batch_size):
    indices = np.random.choice(len(buffer), batch_size, replace=False)
    return [buffer[idx] for idx in indices]
        


def main():
    
    # Intialize players
    agent = DQNPlayer(name="Magnus")
    agent.initialize_model(NROWS,NCOLS,NPLAYERS)
    opponents = [RandomPlayer(name=f"RandomBot") for i in range(NCOLS)]
    opponents += [ColumnSpammer(name=f"CS-{i}",col_preference=i) for i in range(NCOLS)]


    experience_buffer = deque(maxlen=REPLAY_SIZE)
    reward_buffer = deque(maxlen=REWARD_BUFFER_SIZE)
    reward_buffer_vs = {}
    for opp in opponents:
        reward_buffer_vs[opp.name] = deque(maxlen=REWARD_BUFFER_SIZE//len(opponents))

    mse_loss=MeanSquaredError()
    optimizer= Adam(learning_rate=LEARNING_RATE)
    agent.model.summary()

    #writer = SummaryWriter()
    best_reward = 0
    for frame_idx, (transition, q_value, opponent) in enumerate(generate_transitions_pg(agent, opponents)):

        print(f"\nStep {frame_idx}")
        print(transition)
        print(f"q-value: {q_value}")
        print(opponent.name)

        if frame_idx > 10:
            return
        else:
            continue

        

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

            if smoothed_reward > max(SAVE_MODEL_ABS_THRESHOLD,best_reward+SAVE_MODEL_REL_THRESHOLD):
                agent.model.save("magnus.keras")
                best_reward = smoothed_reward

        # Don't start training the network until we have enough data
        if len(experience_buffer) < REPLAY_START_SIZE:
            continue

        training_data = sample_experience_buffer(experience_buffer,BATCH_SIZE)

        
        
        # Compute MSE loss based on chosen move values only
        with tf.GradientTape() as tape:
            pass
            # propagate through NN
 
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
