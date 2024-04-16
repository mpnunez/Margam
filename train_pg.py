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
from connect4lib.pg_player import PolicyPlayer
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


def main():
    
    # Intialize players
    agent = PolicyPlayer(name="Magnus-PG")
    agent.initialize_model(NROWS,NCOLS,NPLAYERS)
    agent.model.summary()

    opponents = [RandomPlayer(name=f"RandomBot") for i in range(NCOLS)]
    opponents += [ColumnSpammer(name=f"CS-{i}",col_preference=i) for i in range(NCOLS)]

    reward_buffer = deque(maxlen=REWARD_BUFFER_SIZE)
    reward_buffer_vs = {}
    for opp in opponents:
        reward_buffer_vs[opp.name] = deque(maxlen=REWARD_BUFFER_SIZE//len(opponents))

    optimizer= Adam(learning_rate=LEARNING_RATE)
    
    

    batch_states = []
    batch_actions = []
    batch_scales = []
    n_episodes_in_batch = 0

    writer = SummaryWriter()
    best_reward = 0
    for frame_idx, (transition, q_value, opponent) in enumerate(generate_transitions_pg(agent, opponents)):

        batch_states.append(transition.board_state)
        batch_actions.append(transition.selected_move)
        batch_scales.append(q_value)

        # Compute average reward
        if transition.resulting_state is None:
            n_episodes_in_batch += 1
            reward_buffer.append(transition.reward)
            reward_buffer_vs[opponent.name].append(transition.reward)
            smoothed_reward = sum(reward_buffer) / len(reward_buffer)
            move_distribution = batch_actions
            move_distribution = np.array([move_distribution.count(i) for i in range(7)])
            move_distribution = move_distribution / move_distribution.sum()
            #print(f"Move distribution: {move_distribution}")
            writer.add_scalar("Average reward", smoothed_reward, frame_idx)
            for opp_name, opp_buffer in reward_buffer_vs.items():
                reward_vs = sum(opp_buffer) / len(opp_buffer) if len(opp_buffer) else 0
                writer.add_scalar(f"reward-vs-{opp_name}", reward_vs, frame_idx)

            if len(reward_buffer) == REWARD_BUFFER_SIZE and smoothed_reward > max(SAVE_MODEL_ABS_THRESHOLD,best_reward+SAVE_MODEL_REL_THRESHOLD):
                agent.model.save(f"{agent.name}-{smoothed_reward}.keras")
                best_reward = smoothed_reward

        # Don't start training the network until we have enough data
        if n_episodes_in_batch < BATCH_N_EPISODES:
            continue

        
        # Chosen moves
        selected_move_mask = one_hot(batch_actions, NCOLS)
        x_train = np.array(batch_states)
        x_train = x_train.swapaxes(1,2).swapaxes(2,3)

        batch_scales = np.array(batch_scales).astype('float32')

        with tf.GradientTape() as tape:

            logits, state_values = agent.model(x_train)

            # State stuff
            #obs_advantage = batch_scales -  state_values
            obs_advantage = batch_scales       # use 0 as baseline value

            # Compute logits
            move_log_probs = tf.nn.log_softmax(logits)
            masked_log_probs = tf.multiply(move_log_probs,selected_move_mask)
            selected_log_probs = tf.reduce_sum(masked_log_probs, 1)
            expectation_loss = - tf.tensordot(obs_advantage,selected_log_probs,axes=1) / len(selected_log_probs)
            

            # Entropy component of loss
            move_probs = tf.nn.softmax(logits)
            entropy_components = tf.multiply(move_probs, move_log_probs)
            entropy_each_state = -tf.reduce_sum(entropy_components, 1)
            entropy = tf.reduce_mean(entropy_each_state)
            entropy_loss = -ENTROPY_BETA * entropy
            loss = expectation_loss + entropy_loss
        
        writer.add_scalar("expectation-loss", expectation_loss.numpy(), frame_idx)
        writer.add_scalar("entropy-loss", entropy_loss.numpy(), frame_idx)
        writer.add_scalar("loss", loss.numpy(), frame_idx)

        #grads = tape.gradient(loss,agent.model.trainable_variables)
        #optimizer.apply_gradients(zip(grads, agent.model.trainable_variables))
        grads = tape.gradient(loss,tape.watched_variables())
        optimizer.apply_gradients(zip(grads, tape.watched_variables()))
        
        # calc KL-div
        new_logits_v, _ = agent.model(x_train)
        new_prob_v = tf.nn.softmax(new_logits_v)
        kl_div_v = -np.sum((np.log((new_prob_v / move_probs)) * move_probs), axis=1).mean()
        writer.add_scalar("kl", kl_div_v.item(), frame_idx)

        # Reset sampling
        batch_states = []
        batch_actions = []
        batch_scales = []
        n_episodes_in_batch = 0

    writer.close()

if __name__ == "__main__":
    main()
