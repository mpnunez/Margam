import numpy as np

from typing import List
from dataclasses import dataclass

class Connect4Error(Exception):
    pass

def get_training_and_viewing_state(game,state):
    """
    For tic-tac-toe and Connect4, convert the
    open_spiel tensor to something trainable
    and viewable from current player POV

    state_np_for_conv: state as numpy array that can be
        fed to a convolutional neural network
        Dimensions 0 and 1 are rows and columns
        Dimension 2 are different players
    human_view_state: 2D numpy array with vacancies
        and different player tokens as different integers
    """

    state_as_tensor = state.observation_tensor()
    tensor_shape = game.observation_tensor_shape()
    state_np = np.reshape(np.asarray(state_as_tensor), tensor_shape)
    
    # Remove last element of 1st dimension showing empty spaces
    state_np = state_np[1::-1,:,:]

    # Move players axis last to be the channels for conv net
    state_np_for_cov = np.moveaxis(state_np, 0, -1)
    
    # view as 1 2D matrix with the last row being first
    human_view_state = state_np[0,::-1,:]+2*state_np[1,::-1,:]

    return state_np_for_cov, human_view_state




@dataclass
class Transition:
    """
    Data from the game used to train the agent
    """
    state: = None
    selected_move: int = 0
    reward: float = 0
    next_state = None

def apply_temporal_difference(transitions,reward_discount,n_td=1):
    """
    Use n_td=-1 to discount to the end of the episode
    """
    if n_td == -1:
        n_td = len(transitions)
    if nt_td < 1:
        raise Connect4Error(f"n_td must be >=1. Got {n_td}")
    transitions_td = []
    for i, tr in enumerate(transitions):
        td_tsn = Transition(
            state = tr.state,
            selected_move = tr.selected_move,
            reward = tr.reward,
        )
        
        for j in range(i+1, min( len(transitions), i+n_td) ):
            td_tsn.reward += transitions[j].reward * reward_discount ** (j-i)

        if i + n_td < len(transitions):
            td_tsn.next_state = transitions[i+n_td].state

        transitions_td.append(td_tsn)
    return transitions_td

def generate_episode_transitions(game_type,hp,agent,opponent,player_pos) -> List:
    game = pyspiel.load_game(game_type)
    state = game.new_initial_state()

    agent_transitions = []
    while not state.is_terminal():
        if state.is_chance_node():
            # Sample a chance event outcome.
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
        else:

            current_player = agent if state.current_player() % 2 == player_pos else opponent
            # If the player action is legal, do it. Otherwise, do random
            action = current_player.get_move(game,state)
            if action not in state.legal_actions():
                action = random.choice(state.legal_actions())
            state.apply_action(action)

    agent_transitions = apply_temporal_difference(
        agent_transitions,
        hp["REWARD_DISCOUNT"],
        n_td=hp["N_TD"],
        ):
    return agent_transitions