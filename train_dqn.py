import itertools
import random
from collections import deque
from enum import Enum
from functools import partial
import yaml

import click
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorboardX import SummaryWriter
from tensorflow import keras, one_hot
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from player import MiniMax, Player
from utils import Connect4Exception
import pyspiel
from utils import get_training_and_viewing_state

class DQNPlayer(Player):
    
    def __init__(self,*args,random_weight=0,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = None
        self.random_weight = random_weight
    
    def get_move(self,game, state) -> int:
        if np.random.random() < self.random_weight:
            return random.choice(state.legal_actions())
        
        state_for_cov, human_view_state = get_training_and_viewing_state(game,state)
        q_values = self.model.predict_on_batch(state_for_cov[np.newaxis,:])[0]
        max_q_ind = np.argmax(q_values)
        
        return game.options[max_q_ind]


def sample_experience_buffer(buffer,batch_size):
    indices = np.random.choice(len(buffer), batch_size, replace=False)
    return [buffer[idx] for idx in indices]
        

def initialize_model(game_type,hp,show_model=True):
    game = pyspiel.load_game(game_type)
    state = game.new_initial_state() 
    state_np_for_cov, human_view_state = get_training_and_viewing_state(game,state)
    nn_input = keras.Input(shape=state_np_for_cov.shape)
    
    if game_type == "tic_tac_toe":
        input_flat = layers.Flatten()(nn_input)
        x = layers.Dense(32, activation="relu")(input_flat)
        q_values = layers.Dense(game.num_distinct_actions(), activation="linear")(x)

        # Deuling DQN adds a second column to the neural net that
        # computes state value V(s) and interprets the Q
        # values as advantage of that action in that state
        # Q(s,a) = A(s,a) + V(s)
        # Final output is the same so it is interoperable with vanilla DQN
        if hp["DEULING_DQN"]:
            x_sv = layers.Dense(32, activation="relu")(input_flat)
            sv = layers.Dense(1, activation="linear")(x_sv)
            q_values = q_values - tf.math.reduce_mean(q_values,axis=1,keepdims=True) + sv
    elif game_type == "connect_four":
        # build some conv net
        raise Connect4Error(f"Need to implement {game_type}")
    else:
        raise Connect4Error(f"Unrecognized game type: {game_type}")

    model = keras.Model(
        inputs=nn_input,
        outputs=q_values,
        name="DQN-model")
    if show_model:
        agent.model.summary()
    return model



def train_dqn(game_type,hp):
    
    # Intialize agent and model
    agent = DQNPlayer(name="Magnus")
    agent.model = initialize_model(game_type,hp)
    target_network = keras.models.clone_model(agent.model)
    target_network.set_weights(agent.model.get_weights())
    return

    game = pyspiel.load_game(game_type)
    opponents = [MiniMax(name="Minnie",max_depth=1)]

    experience_buffer = deque(maxlen=REPLAY_SIZE)
    reward_buffer = deque(maxlen=REWARD_BUFFER_SIZE)
    reward_buffer_vs = {}
    for opp in opponents:
        reward_buffer_vs[opp.name] = deque(maxlen=REWARD_BUFFER_SIZE//len(opponents))

    mse_loss=MeanSquaredError()
    optimizer= Adam(learning_rate=LEARNING_RATE)
    

    writer = SummaryWriter()
    best_reward = 0
    for step, (transition, opponent) in enumerate(generate_transitions(agent, opponents)):

        experience_buffer.append(transition)

        agent.random_weight = max(EPSILON_FINAL, EPSILON_START - step / EPSILON_DECAY_LAST_FRAME)

        opponent = ??
        player_pos = ??
        agent_transitions = generate_episode_transitions(game_type,hp,agent,opponent,player_pos)

        # Compute average reward
        # This double-counts end states if N_TD>1
        # We have a bunch of transitions that look final,
        # but originally were not. Need to figure out
        # how to differentiate final vs. unrolled final-looking states
        if transition.next_state is None:
            reward_buffer.append(transition.reward)
            reward_buffer_vs[opponent.name].append(transition.reward)
            smoothed_reward = sum(reward_buffer) / len(reward_buffer)
            wins = sum(r == game.max_utility() + for r in reward_buffer) 
            ties = sum(r == 0 for r in reward_buffer)
            losses = sum(r == game.min_utility() for r in reward_buffer)
            assert wins + ties + losses == len(reward_buffer)
            move_distribution = [mr.selected_move for mr in experience_buffer]
            move_distribution = np.array([move_distribution.count(i) for i in range(NOUTPUTS)])
            for i in range(game.num_distinct_actions()):
                f = move_distribution[i] / sum(move_distribution)
                writer.add_scalar(f"Action frequency: {i}", f, step)
            move_distribution = move_distribution / move_distribution.sum()
            #print(f"Move distribution: {move_distribution}")
            writer.add_scalar("Average reward", smoothed_reward, step)
            writer.add_scalar("Win rate", wins / len(reward_buffer), step)
            writer.add_scalar("Tie rate", ties / len(reward_buffer), step)
            writer.add_scalar("Loss rate", losses / len(reward_buffer), step)
            for opp_name, opp_buffer in reward_buffer_vs.items():
                reward_vs = sum(opp_buffer) / len(opp_buffer) if len(opp_buffer) else 0
                writer.add_scalar(f"reward-vs-{opp_name}", reward_vs, step)
            if agent.random_weight > EPSILON_FINAL:
                writer.add_scalar("epsilon", agent.random_weight, step)

            if len(reward_buffer) == REWARD_BUFFER_SIZE and smoothed_reward > max(SAVE_MODEL_ABS_THRESHOLD,best_reward+SAVE_MODEL_REL_THRESHOLD):
                agent.model.save(f"magnus-DQN.keras")
                best_reward = smoothed_reward

        # Don't start training the network until we have enough data
        if len(experience_buffer) < REPLAY_START_SIZE:
            continue

        training_data = sample_experience_buffer(experience_buffer,BATCH_SIZE)
        if symmetry:
            training_data = list(itertools.chain(*[g.get_symmetric_transitions(mr) for mr in training_data]))

        # Make X and Y
        x_train = np.array([mr.state for mr in training_data])
        

        # Bellman equation part
        # Take maximum Q(s',a') of board states we end up in
        non_terminal_states = np.array([mr.next_state is not None for mr in training_data])
        resulting_boards = np.array([mr.next_state if mr.next_state is not None else np.zeros(transition.state.shape) for mr in training_data])
        
        # Double DQN - Use on policy network to choose best move
        #   and target network to evaluate the Q-value
        # Single DQN - Take max target network Q to be max Q
        resulting_board_q_target = target_network.predict_on_batch(resulting_boards)
        if double_dqn:
            resulting_board_q_on_policy = agent.model.predict_on_batch(resulting_boards)
            max_move_inds_on_policy = resulting_board_q_on_policy.argmax(axis=1)
            on_policy_move_mask = tf.one_hot(max_move_inds_on_policy,depth=NOUTPUTS)
            target_qs_for_on_policy_moves = tf.multiply(resulting_board_q_target, on_policy_move_mask)
            max_qs = tf.reduce_sum(target_qs_for_on_policy_moves, 1)
        else:
            max_qs = np.max(resulting_board_q_target,axis=1)

        rewards = np.array([mr.reward for mr in training_data])
        q_to_train_single_values = rewards + (DISCOUNT_RATE**N_TD) * np.multiply(non_terminal_states,max_qs)

        # Needed for our mask
        selected_moves = [mr.selected_move for mr in training_data]
        selected_move_mask = one_hot(selected_moves, NOUTPUTS)
        
        # Compute MSE loss based on chosen move values only
        with tf.GradientTape() as tape:
            predicted_q_values = agent.model.predict_on_batch(x_train)
            predicted_q_values = tf.multiply(predicted_q_values,selected_move_mask)
            predicted_q_values = tf.reduce_sum(predicted_q_values, 1)
            q_prediction_errors = predicted_q_values - q_to_train_single_values
            loss_value = mse_loss(q_to_train_single_values, predicted_q_values)
 

        weights_and_biases_flat_before_update = np.concatenate([v.numpy().flatten() for v in agent.model.variables])

        grads = tape.gradient(loss_value, agent.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, agent.model.trainable_variables))

        # Record prediction error
        writer.add_scalar("loss", loss_value.numpy(), step)
        if step % RECORD_HISTOGRAMS == 0:
            writer.add_histogram("q-predicted",predicted_q_values.numpy(),step)
            writer.add_histogram("q-train",q_to_train_single_values,step)
            writer.add_histogram("q-error",q_prediction_errors.numpy(),step)

            weights_and_biases_flat = np.concatenate([v.numpy().flatten() for v in agent.model.variables])
            writer.add_histogram("weights and biases",weights_and_biases_flat,step)

            # Track gradient variance
            grads_flat = np.concatenate([v.numpy().flatten() for v in grads])
            writer.add_histogram("gradients",grads_flat,step)
            grad_rmse = np.sqrt( np.mean( grads_flat ** 2 ) )
            writer.add_scalar("grad_rmse", grad_rmse, step)
            grad_max = np.abs(grads_flat).max()
            writer.add_scalar("grad_max", grad_max, step)

            weights_and_biases_delta = weights_and_biases_flat - weights_and_biases_flat_before_update
            writer.add_histogram("weight-bias-updates",weights_and_biases_delta,step)



        # Update policy
        if step % SYNC_TARGET_NETWORK == 0:

            weights_and_biases_flat = np.concatenate([v.numpy().flatten() for v in agent.model.variables])
            weights_and_biases_target = np.concatenate([v.numpy().flatten() for v in target_network.variables])
            target_parameter_updates = weights_and_biases_flat - weights_and_biases_target
            writer.add_histogram("target parameter updates",target_parameter_updates,step)

            target_network.set_weights(agent.model.get_weights())

    writer.close()

if __name__ == "__main__":
    main()
