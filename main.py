from game import Game
from player import HumanPlayer, RandomPlayer

def main():
    g = Game()
    g.players = [HumanPlayer(),RandomPlayer()]
    g.play_game(show_board_each_move=True)

if __name__ == "__main__":
    main()

