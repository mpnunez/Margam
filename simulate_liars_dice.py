import math
from random import random, randint

from merlob_bot import MerlobBot
from marcel_bot import MarcelBot

def initialize_state_for_bot():
    state = [1, 0]
    for i in range(5):
        die = [0, 0, 0, 0, 0, 0]
        die[randint(0, 5)] = 1
        state += die.copy()
    state += [
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0
    ]
    return state

def interpret_move(move):
    if move == 60:
        return "CALL"
    return [1 + math.floor(move / 6), 1 + move % 6]

def is_valid_move(move, last_move):
    return last_move is None or move > last_move

def is_last_bet_true(last_bet, overall_dice_count):
    num_of_bet_dice, die_face = interpret_move(last_bet)
    return overall_dice_count[die_face - 1] >= num_of_bet_dice

def get_dice_count(dice):
    dice_count = [0, 0, 0, 0, 0, 0]
    for i in range(len(dice)):
        dice_count[dice[i] - 1] += 1
    return dice_count

def simulate_liars_dice(bot1, bot2):
    total_rounds = 10000
    CALL = 60

    bot1_win = 0
    bot2_win = 0
    p1wentfirst = 0
    p2wentfirst = 0
    caller_wins = 0
    for round_number in range(total_rounds):
        bot1_state = initialize_state_for_bot()
        bot2_state = initialize_state_for_bot()
        dice_count1 = get_dice_count(bot1_state)
        dice_count2 = get_dice_count(bot2_state)
        overall_dice_count = []
        for i in range(6):
            overall_dice_count.append(dice_count1[i] + dice_count2[i])

        all_moves = []
        move_count = 0
        bot1_move = 0
        bot2_move = 0
        last_bet = None
        while True:
            assert move_count != 8

            # bot1 goes first half the time
            if (round_number < (total_rounds / 2)) or move_count != 0:
                bot1_move = bot1.play_move(bot1_state)
                # all_moves.append(interpret_move(bot1_move))
                if bot1_move == CALL:
                    # print(round_number, all_moves, is_last_bet_true(last_bet, overall_dice_count))
                    if is_last_bet_true(last_bet, overall_dice_count):
                        bot2_win += 1
                    else:
                        caller_wins += 1
                        bot1_win += 1
                    break
                if not is_valid_move(bot1_move, last_bet):
                    raise Exception("bot1_move was invalid")
                move_count += 1
                last_bet = bot1_move
                bot2_state[32 + bot1_move] = 1


            bot2_move = bot2.play_move(bot2_state)
            # all_moves.append(interpret_move(bot2_move))
            if bot2_move == CALL:
                # print(round_number, all_moves, is_last_bet_true(last_bet, overall_dice_count))
                if is_last_bet_true(last_bet, overall_dice_count):
                    bot1_win += 1
                else:
                    caller_wins += 1
                    bot2_win += 1
                break
            if not is_valid_move(bot2_move, last_bet):
                raise Exception("bot2_move was invalid")
            move_count += 1
            last_bet = bot2_move
            bot1_state[32 + bot2_move] = 1

    print("Caller won", caller_wins/total_rounds)
    print("Merlob v. Marcel", [bot1_win/total_rounds, bot2_win/total_rounds])


def main():
    merlobBot = MerlobBot()
    marcelBot = MerlobBot()
    simulate_liars_dice(merlobBot, marcelBot)

if __name__ == "__main__":
    main()
