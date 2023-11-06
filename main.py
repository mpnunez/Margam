from game import Game
from player import HumanPlayer, RandomPlayer

def main():
    g = Game()
    g.players = [RandomPlayer(),RandomPlayer()]
    records = g.play_game(show_board_each_move=True)
    
    for mr in records:
        mr.assign_scores()
        print(mr)

if __name__ == "__main__":
    main()

