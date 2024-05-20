from tqdm import tqdm
import numpy as np
from enum import Enum
import itertools
from collections import deque

import tensorflow as tf
from tensorflow import one_hot
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorboardX import SummaryWriter
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.nn import softmax

from connect4lib.game import TicTacToe
from connect4lib.agents import RandomPlayer, ColumnSpammer
from connect4lib.agents import MiniMax

from connect4lib.agents.player import Player
import numpy as np
import random



import click

# Game hyperparameters

GAME_TYPE = "TicTacToe"
#GAME_TYPE = "Connect4"

if GAME_TYPE == "Connect4":
    NROWS = 6
    NCOLS = 7
    NPLAYERS = 2
    NCONNECT = 4
    NOUTPUTS = NCOLS

    # Learning
    DISCOUNT_RATE = 0.97
    LEARNING_RATE = 1e-3

    # Recording progress
    REWARD_BUFFER_SIZE = 1_000
    RECORD_HISTOGRAMS = 1_000
    SAVE_MODEL_ABS_THRESHOLD = 0.20
    SAVE_MODEL_REL_THRESHOLD = 0.01

    # Policy gradient
    BATCH_N_EPISODES = 4
    ENTROPY_BETA = 0.1
    STATE_VALUE_BETA = 1

elif GAME_TYPE == "TicTacToe":
    NROWS = 3
    NCOLS = 3
    NPLAYERS = 2
    NCONNECT = 3
    NOUTPUTS = NROWS*NCOLS

    # Learning
    DISCOUNT_RATE = 0.97
    LEARNING_RATE = 1e-4

    # Recording progress
    REWARD_BUFFER_SIZE = 1_000
    RECORD_HISTOGRAMS = 1_000
    SAVE_MODEL_ABS_THRESHOLD = 0
    SAVE_MODEL_REL_THRESHOLD = 0.01

    # Policy gradient
    BATCH_N_EPISODES = 4
    ENTROPY_BETA = 0.1




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

# Merge this together with the n-step TD learning sampling function
# That you used in DQN. Essentially you're unrolling the TD
# and assigning a later final state and reward
def generate_transitions_pg(agent, opponents):
    """
    Sample a full episode, then assign q values
    based on final reward
    """

    episode_transitions = []
    q_values = []
    for transition, opponent in generate_transitions(agent,opponents):

        episode_transitions.append(transition)

        if transition.next_state is not None:
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

@click.command()
@click.option("--symmetry","-s",
    is_flag=True,
    default=False,
    help="Include symmetries in training")
@click.option('-g', '--game-type',
    type=click.Choice(['tictactoe', 'connect4'],
    case_sensitive=False),
    default="tictactoe",
    show_default=True,
    help="game type")
@click.option("--actor-critic","-a",
    is_flag=True,
    default=False,
    help="Use double DQN")
def main(symmetry,game_type,actor_critic):
    
    # Intialize players
    agent = PolicyPlayer(name="VanillaPG")
    input_shape = (NROWS,NCOLS,NPLAYERS)
    nn_input = keras.Input(shape=input_shape)
    
    input_flat = layers.Flatten()(nn_input)
    model_trunk_f  = input_flat

    x = layers.Dense(32, activation="relu")(model_trunk_f)
    logits_output = layers.Dense(NOUTPUTS, activation="linear")(x)
    nn_outputs = logits_output

    if actor_critic:
        x = layers.Dense(64, activation="relu")(model_trunk_f)
        state_value_output = layers.Dense(1, activation="linear")(x)
        nn_outputs = [logits_output,state_value_output]

    agent.model = keras.Model(inputs=nn_input, outputs=nn_outputs, name="policy-model")

    agent.model.summary()

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

        # Compute average reward
        if transition.next_state is None:
            n_episodes_in_batch += 1
            reward_buffer.append(transition.reward)
            reward_buffer_vs[opponent.name].append(transition.reward)
            smoothed_reward = sum(reward_buffer) / len(reward_buffer)
            move_distribution = batch_actions
            move_distribution = np.array([move_distribution.count(i) for i in range(7)])
            move_distribution = move_distribution / move_distribution.sum()
            writer.add_scalar("Average reward", smoothed_reward, step)
            writer.add_scalar("Win rate", (smoothed_reward+1)/2, step)
            for opp_name, opp_buffer in reward_buffer_vs.items():
                reward_vs = sum(opp_buffer) / len(opp_buffer) if len(opp_buffer) else 0
                writer.add_scalar(f"reward-vs-{opp_name}", reward_vs, step)

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
