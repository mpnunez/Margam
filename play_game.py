from tqdm import tqdm
import numpy as np
from enum import Enum
import itertools
from collections import deque
from functools import partial

from connect4lib.game import TicTacToe, Connect4
from connect4lib.agents import RandomPlayer, HumanPlayer, MiniMax, ReinforcePlayer

from keras.models import load_model

import click

game_classes = {
    "tictactoe": TicTacToe,
    "connect4": Connect4,
}
oppenents = {
    "minimax": MiniMax(name="Maximum",max_depth=2),
    "pg": ReinforcePlayer(name="Rein"),
}

@click.command()
@click.option('-g', '--game-type',
    type=click.Choice(['tictactoe', 'connect4'],
    case_sensitive=False),
    default="tictactoe",
    show_default=True)
@click.option('-o', '--opponent',
    type=click.Choice(['minimax', 'pg'],
    case_sensitive=False),
    default="pg",
    show_default=True)
def main(game_type,opponent):
    # Enable choosing opponent with CLI
    
    # Intialize players
    agent = HumanPlayer(name="Marcel")
    opponent = ReinforcePlayer(name="Rein")
    opponent.model = load_model("reinforce-tic.keras")
    g = game_classes[game_type.lower()]()
    g.players = [opponent,agent]
    g.verbose = True
    winner, records = g.play_game()

if __name__ == "__main__":
    main()
