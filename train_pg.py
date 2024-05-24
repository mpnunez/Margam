import itertools
import random
from collections import deque
from enum import Enum

import click
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorboardX import SummaryWriter
from tensorflow import keras, one_hot
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.nn import softmax
from tqdm import tqdm

from player import ColumnSpammer, MiniMax, Player, RandomPlayer
from utils import Connect4Error


class PolicyPlayer(Player):
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = None
    
    def get_move(self,board: np.array, game) -> int:
        logits = self.model.predict_on_batch(board[np.newaxis,:])
        if len(self.model.outputs) == 2:        # for actor-critic
            logits, _ = logits
        move_probabilities = softmax(logits[0])
        selected_move = random.choices(game.options, weights=move_probabilities, k=1)[0]
        return selected_move

def initialize_model(game_type,hp,show_model=True):
    game = pyspiel.load_game(game_type)
    state = game.new_initial_state() 
    state_np_for_cov, human_view_state = get_training_and_viewing_state(game,state)
    nn_input = keras.Input(shape=state_np_for_cov.shape)
    
    if game_type == "tic_tac_toe":
        input_flat = layers.Flatten()(nn_input)
        model_trunk_f  = input_flat

        x = layers.Dense(32, activation="relu")(model_trunk_f)
        logits_output = layers.Dense(game.num_distinct_actions(), activation="linear")(x)
        nn_outputs = logits_output

        if hp["ACTOR_CRITIC"]:
            x = layers.Dense(64, activation="relu")(model_trunk_f)
            state_value_output = layers.Dense(1, activation="linear")(x)
            nn_outputs = [logits_output,state_value_output]

    else:
        raise Connect4Error(f"{game_type} not implemented")

    agent.model = keras.Model(inputs=nn_input, outputs=nn_outputs, name="policy-model")

    if show_model:
        agent.model.summary()
    

def train_pg(game_type,hp):
    
    # Cannot do tempral differencing without critic
    if not hp["ACTOR_CRITIC"]:
        hp["N_TD"] = -1

    # Intialize players
    agent = PolicyPlayer(name="VanillaPG")
    agent.model = initialize_model(game_type,hp)

    

    opponents = [MiniMax(name="Minnie",max_depth=1)]
    reward_buffer = deque(maxlen=REWARD_BUFFER_SIZE)
    reward_buffer_vs = {}
    for opp in opponents:
        reward_buffer_vs[opp.name] = deque(maxlen=REWARD_BUFFER_SIZE//len(opponents))

    optimizer= Adam(learning_rate=LEARNING_RATE)
    mse_loss=MeanSquaredError()
    

    batch_states = []
    batch_actions = []
    batch_scales = []
    n_episodes_in_batch = 0

    writer = SummaryWriter()
    best_reward = 0
    for step, (transition, q_value, opponent) in enumerate(generate_transitions_pg(agent, opponents)):

        batch_states.append(transition.state)
        batch_actions.append(transition.selected_move)
        batch_scales.append(q_value)

        opponent = opponents[(episode_ind//2)%len(opponents)]
        player_pos = episode_ind%2
        agent_transitions = generate_episode_transitions(game_type,hp,agent,opponent,player_pos)
        episode_ind += 1

        reward_buffer.append(agent_transitions[-1].reward)
        opp_buffer[opponent.name].append(agent_transitions[-1])
        record_episode_statistics(
            writer,
            game,
            step,
            experience_buffer,
            reward_buffer,
            reward_buffer_vs
        )

        if len(reward_buffer) == REWARD_BUFFER_SIZE and smoothed_reward > max(SAVE_MODEL_ABS_THRESHOLD,best_reward+SAVE_MODEL_REL_THRESHOLD):
            agent.model.save(f"{agent.name}.keras")
            best_reward = smoothed_reward

        # Don't start training the network until we have enough data
        if n_episodes_in_batch < BATCH_N_EPISODES:
            continue

        
        # Chosen moves
        selected_move_mask = one_hot(batch_actions, NOUTPUTS)
        x_train = np.array(batch_states)
        

        batch_scales = np.array(batch_scales).astype('float32')

        with tf.GradientTape() as tape:

            logits = agent.model.predict_on_batch(x_train)
            if actor_critic:
                logits, state_values = logits
                state_values = state_values[:,0]
                state_loss = mse_loss(batch_scales, state_values)
            
            # Compute logits
            move_log_probs = tf.nn.log_softmax(logits)
            masked_log_probs = tf.multiply(move_log_probs,selected_move_mask)
            selected_log_probs = tf.reduce_sum(masked_log_probs, 1)
            obs_advantage = batch_scales
            if actor_critic:
                obs_advantage = batch_scales -  tf.stop_gradient(state_values)
            expectation_loss = - tf.tensordot(obs_advantage,selected_log_probs,axes=1) / len(selected_log_probs)
            

            # Entropy component of loss
            move_probs = tf.nn.softmax(logits)
            entropy_components = tf.multiply(move_probs, move_log_probs)
            entropy_each_state = -tf.reduce_sum(entropy_components, 1)
            entropy = tf.reduce_mean(entropy_each_state)
            entropy_loss = -ENTROPY_BETA * entropy

            # Sum the loss contributions
            loss = expectation_loss + entropy_loss
            if actor_critic:
                loss += state_loss
                writer.add_scalar("state-loss", state_loss.numpy(), step)

        writer.add_scalar("expectation-loss", expectation_loss.numpy(), step)
        writer.add_scalar("entropy-loss", entropy_loss.numpy(), step)
        writer.add_scalar("loss", loss.numpy(), step)

        grads = tape.gradient(loss,agent.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, agent.model.trainable_variables))
        #grads = tape.gradient(loss,tape.watched_variables())
        #optimizer.apply_gradients(zip(grads, tape.watched_variables()))
        
        # calc KL-div
        new_logits_v = agent.model.predict_on_batch(x_train)
        if actor_critic:
            new_logits_v, _ = new_logits_v
        new_prob_v = tf.nn.softmax(new_logits_v)
        KL_EPSILON = 1e-7
        new_prob_v_kl = new_prob_v + KL_EPSILON
        move_probs_kl = move_probs + KL_EPSILON
        kl_div_v = -np.sum(np.log(new_prob_v_kl / move_probs_kl) * move_probs_kl, axis=1).mean()
        writer.add_scalar("kl", kl_div_v.item(), step)

        # Track gradient variance
        weights_and_biases_flat = np.concatenate([v.numpy().flatten() for v in agent.model.variables])
        writer.add_histogram("weights and biases",weights_and_biases_flat,step)
        grads_flat = np.concatenate([v.numpy().flatten() for v in grads])
        writer.add_histogram("gradients",grads_flat,step)
        grad_rmse = np.sqrt( np.mean( grads_flat ** 2 ) )
        writer.add_scalar("grad_rmse", grad_rmse, step)
        grad_max = np.abs(grads_flat).max()
        writer.add_scalar("grad_max", grad_max, step)

        # Reset sampling
        batch_states = []
        batch_actions = []
        batch_scales = []
        n_episodes_in_batch = 0

    writer.close()

if __name__ == "__main__":
    main()
