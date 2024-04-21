from tqdm import tqdm
import numpy as np
from enum import Enum
import itertools
from collections import deque
from functools import partial

from connect4lib.game.tictactoe import TicTacToe
from connect4lib.agents.player import RandomPlayer, HumanPlayer


def main():
    
    # Intialize players
    agent = HumanPlayer(name="Marcel")
    opponent = RandomPlayer(name="Random")
    g = TicTacToe()
    g.players = [agent, opponent]
    g.verbose = True
    winner, records = g.play_game()

if __name__ == "__main__":
    main()
