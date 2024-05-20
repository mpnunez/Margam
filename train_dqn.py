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

from player import Player, MiniMax
import numpy as np
import random


from tensorflow import keras
from tensorflow.keras import layers

import click

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

    # DQN
    BATCH_SIZE = 32             
    REPLAY_SIZE = 10_000
    SYNC_TARGET_NETWORK = 1_000
    REPLAY_START_SIZE = 10_000
    EPSILON_DECAY_LAST_FRAME = 5e4
    EPSILON_START = 1.0
    EPSILON_FINAL = 0.02

elif GAME_TYPE == "TicTacToe":
    NROWS = 3
    NCOLS = 3
    NPLAYERS = 2
    NCONNECT = 3
    NOUTPUTS = NROWS*NCOLS

    # Learning
    DISCOUNT_RATE = 0.90
    LEARNING_RATE = 1e-4

    # Recording progress
    REWARD_BUFFER_SIZE = 1_000
    RECORD_HISTOGRAMS = 1_000
    SAVE_MODEL_ABS_THRESHOLD = 0
    SAVE_MODEL_REL_THRESHOLD = 0.01

    # DQN
    BATCH_SIZE = 32             
    REPLAY_SIZE = 10_000
    SYNC_TARGET_NETWORK = 1_000
    REPLAY_START_SIZE = 10_000
    EPSILON_DECAY_LAST_FRAME = 1e5
    EPSILON_START = 1.0
    EPSILON_FINAL = 0.02

N_TD = 2        # temporal difference learning look-ahead

class DQNPlayer(Player):
    
    def __init__(self,*args,random_weight=0,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = None
        self.random_weight = random_weight
    
    def get_move(self,board: np.array, game) -> int:
        if np.random.random() < self.random_weight:
            return random.choice(game.options)
            
        q_values = self.model.predict_on_batch(board[np.newaxis,:])[0]
        max_q_ind = np.argmax(q_values)
        
        return game.options[max_q_ind]


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

        agent_records_td = []
        for i, tr in enumerate(agent_records):
            td_tsn = Transition(
                state = tr.state,
                selected_move = tr.selected_move,
                reward = tr.reward,
            )
            
            for j in range(i+1, min( len(agent_records), i+N_TD) ):
                td_tsn.reward += agent_records[j].reward * DISCOUNT_RATE ** (j-i)

            if i + N_TD < len(agent_records):
                td_tsn.next_state = agent_records[i+N_TD].state

            agent_records_td.append(td_tsn)

        for move_record in agent_records_td:
            yield move_record, opponent





def sample_experience_buffer(buffer,batch_size):
    indices = np.random.choice(len(buffer), batch_size, replace=False)
    return [buffer[idx] for idx in indices]
        


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
@click.option("--double-dqn","-d",
    is_flag=True,
    default=True,
    help="Use double DQN")
@click.option("--deuling-dqn","-u",
    is_flag=True,
    default=True,
    help="Use deuling DQN")
def main(symmetry,game_type,double_dqn,deuling_dqn):
    
    # Intialize model
    agent = DQNPlayer(name="Magnus")
    input_shape = (NROWS,NCOLS,NPLAYERS)
    nn_input = keras.Input(shape=input_shape)
    input_flat = layers.Flatten()(nn_input)
    x = layers.Dense(32, activation="relu")(input_flat)
    #x = layers.Dense(16, activation="relu")(x)
    q_values = layers.Dense(NOUTPUTS, activation="linear")(x)

    # Deuling DQN adds a second column to the neural net that
    # computes state value V(s) and interprets the Q
    # values as advantage of that action in that state
    # Q(s,a) = A(s,a) + V(s)
    # Final output is the same so it is interoperable with vanilla DQN
    if deuling_dqn:
        x_sv = layers.Dense(32, activation="relu")(input_flat)
        sv = layers.Dense(1, activation="linear")(x_sv)
        q_values = q_values - tf.math.reduce_mean(q_values,axis=1,keepdims=True) + sv

    agent.model = keras.Model(
        inputs=nn_input,
        outputs=q_values,
        name="DQN-model")
    agent.model.summary()
    print(len(agent.model.outputs))
    return
    target_network = keras.models.clone_model(agent.model)
    target_network.set_weights(agent.model.get_weights())

    g = TicTacToe(nrows=NROWS,ncols=NCOLS,nconnectwins=NCONNECT)
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

        # Compute average reward
        # This double-counts end states if N_TD>1
        # We have a bunch of transitions that look final,
        # but originally were not. Need to figure out
        # how to differentiate final vs. unrolled final-looking states
        if transition.next_state is None:
            reward_buffer.append(transition.reward)
            reward_buffer_vs[opponent.name].append(transition.reward)
            smoothed_reward = sum(reward_buffer) / len(reward_buffer)
            win_rate = sum(r == g.WIN_REWARD for r in reward_buffer) / len(reward_buffer)
            tie_rate = sum(r == g.TIE_REWARD for r in reward_buffer) / len(reward_buffer)
            loss_rate = sum(r == g.LOSS_REWARD for r in reward_buffer) / len(reward_buffer)
            move_distribution = [mr.selected_move for mr in experience_buffer]
            move_distribution = np.array([move_distribution.count(i) for i in range(NOUTPUTS)])
            for i in range(NOUTPUTS):
                f = move_distribution[i] / sum(move_distribution)
                writer.add_scalar(f"Fraction choice {i}", f, step)
            move_distribution = move_distribution / move_distribution.sum()
            #print(f"Move distribution: {move_distribution}")
            writer.add_scalar("Average reward", smoothed_reward, step)
            writer.add_scalar("Win rate", win_rate, step)
            writer.add_scalar("Tie rate", tie_rate, step)
            writer.add_scalar("Loss rate", loss_rate, step)
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
