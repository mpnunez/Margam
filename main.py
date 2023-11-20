from game import Game
from player import RandomPlayer
from aiplayer import AIPlayer
from tqdm import tqdm

def play_matches(player1, player2, n_games=100):
    all_move_records = []
    winners = []
    for _ in tqdm(range(n_games)):
        #print(f"{i}/{n_training_games}")
        g = Game()
        #g.verbose= True
        g.players = [player1,player2]
        winner, records = g.play_game()
        winners.append(winner)
        
        for mr in records:
            mr.assign_scores()
 
        all_move_records += records
        
    win_loss_ties= {
        "wins": sum(w == 0 for w in winners) / len(winners),
        "loss": sum(w == 1 for w in winners) / len(winners),
        "tie": sum(w == -1 for w in winners) / len(winners),
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

