from game import Game
from player import RandomPlayer
from aiplayer import AIPlayer

def main():
    
    n_training_games = 10
    magnus = AIPlayer(name="Magnus")
    
    all_move_records = []
    for i in range(n_training_games):
        #print(f"{i}/{n_training_games}")
        g = Game()
        g.players = [magnus,RandomPlayer("Random Bot")]
        records = g.play_game(show_board_each_move=False)
        
        for mr in records:
            mr.assign_scores()
            
        
        
            
        all_move_records += records
        
    win_loss_ties= {
        "wins": sum(mr.result == 0 for mr in all_move_records) / len(all_move_records),
        "loss": sum(mr.result == 1 for mr in all_move_records) / len(all_move_records),
        "tie": sum(mr.result == 0.5 for mr in all_move_records) / len(all_move_records),
    }
    print("Before training")
    print(win_loss_ties)
        
    #print(all_move_records[-1])
    magnus.train_on_game_data(all_move_records)
    
    all_move_records = []
    for i in range(n_training_games):
        #print(f"{i}/{n_training_games}")
        g = Game()
        g.players = [magnus,RandomPlayer("Random Bot")]
        records = g.play_game(show_board_each_move=False)
        
        for mr in records:
            mr.assign_scores()
            
        
        
            
        all_move_records += records
        
    win_loss_ties= {
        "wins": sum(mr.result == 0 for mr in all_move_records) / len(all_move_records),
        "loss": sum(mr.result == 1 for mr in all_move_records) / len(all_move_records),
        "tie": sum(mr.result == 0.5 for mr in all_move_records) / len(all_move_records),
    }
    print("After training")
    print(win_loss_ties)

if __name__ == "__main__":
    main()

