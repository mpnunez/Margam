from tqdm import tqdm
import numpy as np
from enum import Enum
import itertools
from collections import deque
from functools import partial

from connect4lib.game import TicTacToe, Connect4
from connect4lib.agents import HumanPlayer, MiniMax, ReinforcePlayer, PolicyPlayer
from train_dqn import 

from keras.models import load_model

import click

import random
import pyspiel
import numpy as np
import random


game_classes = {
    "tictactoe": TicTacToe,
    "connect4": Connect4,
}

@click.command()
@click.option('-g', '--game-type',
    type=click.Choice(['tictactoe', 'connect4'],
    case_sensitive=False),
    default="tictactoe",
    show_default=True,
    help="game type")
@click.option('-o', '--opponent',
    type=click.Choice(['minimax','dqn', 'pg', 'pgac'],
    case_sensitive=False),
    default="pg",
    show_default=True,
    help="opponent type")
@click.option('-d', '--depth',
    type=int,
    default=2,
    show_default=True,
    help="Depth for minimax")
@click.option('-m', '--model',
    type=str,
    default=None,
    help="Model file to load for AI player",
)
@click.option("--second",
    is_flag=True,
    default=False,
    help="Play as second player")
def main(game_type,opponent,depth,model,second):
    # Enable choosing opponent with CLI
    
    # Intialize players
    human = HumanPlayer(name="Marcel")

    if opponent.lower() == "minimax":
        opponent = MiniMax(name="Maximus",max_depth=depth)
    elif opponent.lower() == "pg":
        opponent = ReinforcePlayer(name="PG")
        opponent.model = load_model(model)
        opponent.model.summary()
    elif opponent.lower() == "dqn":
        opponent = DQNPlayer(name="DQN")
        opponent.model = load_model(model)
        opponent.model.summary()
    elif opponent.lower() == "pgac":
        opponent = PolicyACPlayer(name="PGAC")
        opponent.model = load_model(model)
        opponent.model.summary()

    g = game_classes[game_type.lower()]()
    g.players = [opponent,human] if second else [human,opponent]
    g.verbose = True
    winner, records = g.play_game()


    

    game = pyspiel.load_game("connect_four")
    state = game.new_initial_state()
    i = 0
    while not state.is_terminal():
        #if i == 2:
        #    break
        legal_actions = state.legal_actions()
        print(f"\nCurrent player: {state.current_player()+1}")

        print("\nPOV State")
        state_as_tensor = state.observation_tensor()
        tensor_shape = game.observation_tensor_shape()
        state_np = np.reshape(np.asarray(state_as_tensor), tensor_shape)
        state_np = state_np[1::-1,:,:]
        print(state_np)

        # Move players axis last to be the channels
        # for conv net
        state_np_for_cov = np.moveaxis(state_np, 0, -1)
        
        # view as 1 2D matrix with the last row being first
        ind_rep = state_np[0,::-1,:]+2*state_np[1,::-1,:]
        print(ind_rep)
        

        #if i == 1:
        #    break

        if state.is_chance_node():
            # Sample a chance event outcome.
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
        else:
            # The algorithm can pick an action based on an observation (fully observable
            # games) or an information state (information available for that player)
            # We arbitrarily select the first available action as an example.
            action = random.choice(legal_actions)
            print(f"Action: {action}")
            state.apply_action(action)

        i += 1

    print(state.returns())
    print(state.rewards())

if __name__ == "__main__":
    main()
