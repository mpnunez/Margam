from tqdm import tqdm
import numpy as np
from enum import Enum
import itertools
from collections import deque
from functools import partial

from connect4lib.game.tictactoe import TicTacToe
from connect4lib.game.connect4 import Connect4
from connect4lib.agents.player import RandomPlayer, HumanPlayer
from connect4lib.agents.minimax import MiniMax

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
    show_default=True)
def main(game_type):
    # Enable choosing opponent with CLI
    
    # Intialize players
    agent = HumanPlayer(name="Marcel")
    opponent = MiniMax(name="Maximus")
    opponent.max_depth = 3
    g = game_classes[game_type.lower()]()
    g.players = [opponent,agent]
    g.verbose = True
    winner, records = g.play_game()

if __name__ == "__main__":
    main()
