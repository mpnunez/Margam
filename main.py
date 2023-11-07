from game import Game
from player import RandomPlayer
from aiplayer import AIPlayer
from tqdm import tqdm

def play_matches(player1, player2, n_games=100):
    all_move_records = []
    for _ in tqdm(range(n_games)):
        #print(f"{i}/{n_training_games}")
        g = Game()
        g.players = [player1,player2]
        records = g.play_game(show_board_each_move=False)
        
        for mr in records:
            mr.assign_scores()
            
        
        
            
        all_move_records += records
        
    win_loss_ties= {
        "wins": sum(mr.result == 0 for mr in all_move_records) / len(all_move_records),
        "loss": sum(mr.result == 1 for mr in all_move_records) / len(all_move_records),
        "tie": sum(mr.result == 0.5 for mr in all_move_records) / len(all_move_records),
    }
        
    return all_move_records, win_loss_ties
        

def main():
    
    magnus = AIPlayer(name="Magnus")
    random_bot = RandomPlayer("Random Bot")
    
    all_move_records, win_loss_ties = play_matches(random_bot, random_bot, n_games=1000)
    print("Before training")
    print(win_loss_ties)
        
    #print(all_move_records[-1])
    magnus.train_on_game_data(all_move_records)
    
    all_move_records, win_loss_ties = play_matches(magnus, random_bot, n_games=100)
    print("After training")
    print(win_loss_ties)

if __name__ == "__main__":
    main()

