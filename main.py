from game import Game
from player import RandomPlayer
from aiplayer import AIPlayer

def main():
    
    n_training_games = 100
    magnus = AIPlayer()
    
    all_move_records = []
    for i in range(n_training_games):
        print(f"{i}/{n_training_games}")
        g = Game()
        g.players = [magnus,RandomPlayer()]
        records = g.play_game(show_board_each_move=False)
        
        for mr in records:
            mr.assign_scores()
            
        all_move_records += records
        
    print(all_move_records[-1])
    magnus.train_on_game_data(all_move_records)

if __name__ == "__main__":
    main()

