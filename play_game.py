from tqdm import tqdm
import numpy as np
from enum import Enum
import itertools
from collections import deque
from functools import partial

from connect4lib.game import TicTacToe, Connect4
from connect4lib.agents import HumanPlayer, MiniMax, ReinforcePlayer, PolicyPlayer

from keras.models import load_model

import click

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
        opponent = ReinforcePlayer(name="DQN")
        opponent.model = load_model(model)
        opponent.model.summary()
    elif opponent.lower() == "pgac":
        opponent = PolicyPlayer(name="PGAC")
        opponent.model = load_model(model)
        opponent.model.summary()

    g = game_classes[game_type.lower()]()
    g.players = [opponent,human] if second else [human,opponent]
    g.verbose = True
    winner, records = g.play_game()

if __name__ == "__main__":
    main()
