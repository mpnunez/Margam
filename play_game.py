from tqdm import tqdm
import numpy as np
from enum import Enum
import itertools
from collections import deque
from functools import partial

from connect4lib.game.tictactoe import TicTacToe
from connect4lib.agents.player import RandomPlayer, HumanPlayer
from connect4lib.agents.minimax import MiniMax

def main():
    # Enable choosing opponent with CLI
    
    # Intialize players
    agent = HumanPlayer(name="Marcel")
    opponent = MiniMax(name="Maximus")
    opponent.max_depth = 1
    g = TicTacToe()
    g.players = [opponent,agent]
    g.verbose = True
    winner, records = g.play_game()

if __name__ == "__main__":
    main()
