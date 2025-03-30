import math
from margam.player import Player
from margam.rl import GameType, MargamError

CALL = 60

def exactly_k_of_number(n, k):
    """
    n - number of dice
    k - number of elements we want to be the same value
    p - probability of rolling any value from a die: set as constant 1/6
    """
    p = 1/6
    return math.comb(n, k) * math.pow(p, k) * math.pow(1 - p, n - k)

def get_default_probability_matrix(n):
    probability_matrix = []
    s = 0
    for k in reversed(range(n+1)):
        s += exactly_k_of_number(n, k)
        probability_matrix.append(round(s,4))
    probability_matrix.reverse()
    probability_matrix[0] = 1
    return probability_matrix

def interpret_move(move):
    if move == 60:
        return "CALL"
    return [1 + math.floor(move / 6), 1 + move % 6]

def is_last_bet_true(last_bet, overall_dice_count):
    num_of_bet_dice, die_face = interpret_move(last_bet)
    return overall_dice_count[die_face - 1] >= num_of_bet_dice

# Produces an array of arrays that is:
# [
#    [Probability of 0 more]
#    [Probability of 0-1 more]
#    [Probability of 0-2 more]
#    ...
#    [Probability of 0-n more]
# ]
# def get_computed_probability_matrices(max_n):
#     probability_matrices = [[1.0]]
#     for n in range(1, max_n+1):
#         probability_matrices.append(get_default_probability_matrix(n))
#     return probability_matrices


from random import random, randint

NUMBER_OF_DICE = 10
MAX_BET_INDEX = NUMBER_OF_DICE * 6
EGREGIOUS_LIE = 0.05
HONEST_PROBABILITY_THRESHOLD = 0.50
DEFAULT_PROBABILITY_MATRIX = get_default_probability_matrix(NUMBER_OF_DICE)
PROBABILITY_OF_EGREGIOUS_LYING = 0
NO_BETTER_BID_THRESHOLD = 0.50


def get_last_bet(bets):
    for i in reversed(range(len(bets))):
        if bets[i] == 1:
            return i
    return 0


# Returns an array with the probabilities 0-1 of that many dice where
#     index 0 is the bid (1, 1)
#     index 59 is the bid (10, 6)
def get_probability_matrix(dice_count, num_other_dice, last_bet, bets):
    probability_of_k_more = get_default_probability_matrix(num_other_dice)

    adj_dice_count = dice_count.copy()
    num_of_bet_dice, die_face = interpret_move(last_bet)
    incorporate_last_bet = last_bet != 0 and sum(bets) < 2 and (3 <= num_of_bet_dice <= 4)
    if incorporate_last_bet:
        num_of_suspected_dice = num_of_bet_dice - 1
        adj_dice_count[die_face - 1] += num_of_suspected_dice


    probability_matrix = []
    for i in range(NUMBER_OF_DICE):
        for j in range(6):
            n_have = adj_dice_count[j]
            n_need = (i + 1) - n_have
            if n_need <= 0:
                probability_matrix.append(1)
            elif n_need > num_other_dice:
                probability_matrix.append(0)
            elif incorporate_last_bet and (j + 1) == die_face:
                probability_matrix.append(0)
            else:
                # probability_of_k_more = [1, 0.60, 0.20, 0.04, 0.003, 0.0001] for n = 5
                probability_matrix.append(probability_of_k_more[n_need])
    return probability_matrix

# def adjust_probability_matrix(probability_matrix, last_bet, bets):
#     # We are going to assume only the last bet has any information
#     # and that if they didn't lie, they just added 1 to their count
#     # (conditional on that number never having been previously bet)
#     if last_bet == 0:
#         return probability_matrix

#     num_of_bet_dice, die_face = interpret_move(last_bet)
#     num_of_actual_dice = num_of_bet_dice - 1

#     if sum(bets) < 2:
#         adj_probability_matrix = probability_matrix.copy()
#         for i in range(NUMBER_OF_DICE):
#             index = (i * 6) + (die_face - 1)
#             adj_probability_matrix[index]
#     else:
#         return probability_matrix


# x = get_probability_matrix([0, 6, 0, 0, 0, 0], 5)
# print(x)


# This function should be made more complex if we aren't always having 5 dice
def has_improbable_private_info(dice_count):
    return max(dice_count) >= 3


def call_guarentees_win(probability_matrix, last_bet):
    # if they have 4 (0.003) or 5 (0.0001) of something in their hand,
    # then let them win
    return probability_matrix[last_bet] < 0.01 


def has_no_better_bid(probability_matrix, last_bet):
    for i in range(last_bet + 1, min(last_bet + 7, MAX_BET_INDEX)):
        if probability_matrix[i] > NO_BETTER_BID_THRESHOLD:
            return False
    return True



def bid_honestly(probability_matrix, last_bet):
    threshold = HONEST_PROBABILITY_THRESHOLD
    if random() < 0.25:
        threshold = 1
     
    for i in reversed(range(last_bet + 1, len(probability_matrix))):
        if probability_matrix[i] >= threshold:
            return i
    return CALL

# You don't want to egregiously lie so high that the other person doubts,
# you want to bid high enough that the opponent bids higher, suspecting
# that no bid higher will be valid
def bid_egregious_lie(probability_matrix, last_bet):
    pass


def bid_lie(probability_matrix, last_bet):
    return randint(6, 12)

def get_die(die):
    for i in range(6):
        if die[i] == 1:
            return i + 1
    raise Error("Expected die to include 1")

def get_dice_from_state(dice_state):
    die1 = get_die(dice_state[0:6])
    die2 = get_die(dice_state[6:12])
    die3 = get_die(dice_state[12:18])
    die4 = get_die(dice_state[18:24])
    die5 = get_die(dice_state[24:30])
    return [die1, die2, die3, die4, die5]

def get_dice_count(dice):
    dice_count = [0, 0, 0, 0, 0, 0]
    for i in range(len(dice)):
        dice_count[dice[i] - 1] += 1
    return dice_count

import numpy as np

class MerlobBot(Player):
    def __init__(self, game_handler, name=None, probability_of_lying = 0):
        super().__init__(game_handler, name)
        self.probability_of_lying = probability_of_lying
        if self.game_handler.GameType != GameType.LIARS_DICE:
            raise MargamError("MerlobBot can only play liars_dice.")

    def get_move(self, state) -> int:
        state_as_tensor = state.observation_tensor()
        tensor_shape = self.game_handler.game.observation_tensor_shape()
        state_tensor = np.reshape(np.asarray(state_as_tensor), tensor_shape)
        return self.play_move(state_tensor)

    def play_move(self, state):
        assert len(state) == 93
        # state[0] = 1 if opponent's turn else 0
        # state[1] = 1 if our turn else 0
        dice = get_dice_from_state(state[2:32])
        dice_count = get_dice_count(dice)
        bets = state[32:92]
        last_bet = get_last_bet(bets)
        # state[92] = 1 if last bet was 1
        assert state[92] != 1

        # We return an action: 0 corresponds to one 1, 1 corresponds to one 2, and so on
        # with the final action corresponding to CALL
        CALL = 6 * NUMBER_OF_DICE

        num_unknown_dice = NUMBER_OF_DICE - len(dice)
        probability_matrix = get_probability_matrix(dice_count, num_unknown_dice, last_bet, bets) 
        # for i in range(10):
        #     print(probability_matrix[(i * 6):(i * 6 + 6)])

        if call_guarentees_win(probability_matrix, last_bet):
            return CALL
        elif has_no_better_bid(probability_matrix, last_bet):
            return CALL
        
        if has_improbable_private_info(dice_count):
            return bid_honestly(probability_matrix, last_bet)
        elif sum(bets) == 0:
            return randint(6, 12)
        else:
            r = random()
            if r < PROBABILITY_OF_EGREGIOUS_LYING:
                return bid_egregious_lie(probability_matrix, last_bet)
            elif last_bet < 6 and r < self.probability_of_lying:
                return bid_lie(probability_matrix, last_bet)
            else:
                return bid_honestly(probability_matrix, last_bet)

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


def play_game():
    bot1 = MerlobBot()
    bot2 = MerlobBot()

    bot1_win = 0
    bot2_win = 0
    caller_wins = 0
    for i in range(1000):
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
        while True:
            move_count += 1
            assert move_count != 8
            bot1_move = bot1.play_move(bot1_state)
            all_moves.append(interpret_move(bot1_move))
            if bot1_move == 60:
                last_bet = get_last_bet(bot1_state[32:92])
                # print(all_moves, is_last_bet_true(last_bet, overall_dice_count))
                if is_last_bet_true(last_bet, overall_dice_count):
                    bot2_win += 1
                else:
                    caller_wins += 1
                    bot1_win += 1
                break

            bot2_state[32 + bot1_move] = 1
            bot2_move = bot2.play_move(bot2_state)
            all_moves.append(interpret_move(bot2_move))
            if bot2_move == 60:
                last_bet = get_last_bet(bot2_state[32:92])
                # print(all_moves, is_last_bet_true(last_bet, overall_dice_count))
                if is_last_bet_true(last_bet, overall_dice_count):
                    bot1_win += 1
                else:
                    caller_wins += 1
                    bot2_win += 1
                break

            bot1_state[32 + bot2_move] = 1

    print([bot1_win, bot2_win], caller_wins)

# play_game()


# EXAMPLE_STATE_ONE = [
#     0, # not opponent's turn
#     1, # our turn
#     0, 1, 0, 0, 0, 0, # die_1 is 2 
#     0, 0, 0, 1, 0, 0, # die_2 is 4 
#     0, 0, 1, 0, 0, 0, # die_3 is 3 
#     0, 1, 0, 0, 0, 0, # die_4 is 2 
#     0, 0, 0, 0, 1, 0, # die_5 is 5 
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0, # nothing bid
#     0,    # no call
# ]

# assert play_move(EXAMPLE_STATE_ONE) == 13 # three 2s

# EXAMPLE_STATE_TWO = [
#     0, # not opponent's turn
#     1, # our turn
#     0, 1, 0, 0, 0, 0, # die_1 is 2 
#     0, 1, 0, 0, 0, 0, # die_2 is 2 
#     0, 1, 0, 0, 0, 0, # die_3 is 2 
#     0, 0, 0, 1, 0, 0, # die_4 is 4 
#     0, 0, 0, 0, 1, 0, # die_5 is 5 
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0, # nothing bid
#     0,    # no call
# ]

# assert play_move(EXAMPLE_STATE_TWO) == 19 # four 2s


# EXAMPLE_STATE_THREE = [
#     0, # not opponent's turn
#     1, # our turn
#     1, 0, 0, 0, 0, 0, # die_1 is 1
#     0, 1, 0, 0, 0, 0, # die_2 is 2 
#     0, 1, 0, 0, 0, 0, # die_3 is 2 
#     0, 0, 0, 1, 0, 0, # die_4 is 4 
#     0, 0, 0, 0, 1, 0, # die_5 is 5 
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 1, # three 6s bid
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0,    # no call
# ]

# assert play_move(EXAMPLE_STATE_THREE) == 60 # CALL


# EXAMPLE_STATE_FOUR = [
#     0, # not opponent's turn
#     1, # our turn
#     1, 0, 0, 0, 0, 0, # die_1 is 1
#     0, 1, 0, 0, 0, 0, # die_2 is 2 
#     0, 1, 0, 0, 0, 0, # die_3 is 2 
#     0, 0, 0, 1, 0, 0, # die_4 is 4 
#     0, 0, 0, 0, 1, 0, # die_5 is 5 
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 1, 0, 0, 0, 0, # three 2s bid
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0,    # no call
# ]

# assert play_move(EXAMPLE_STATE_FOUR) == 19 # four 2s



# EXAMPLE_STATE_FIVE = [
#     0, # not opponent's turn
#     1, # our turn
#     1, 0, 0, 0, 0, 0, # die_1 is 1
#     0, 1, 0, 0, 0, 0, # die_2 is 2
#     0, 0, 1, 0, 0, 0, # die_3 is 3 
#     0, 0, 1, 0, 0, 0, # die_4 is 3 
#     0, 0, 0, 0, 1, 0, # die_5 is 5 
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 1, 0, 0, 0, 0, # three 2s bid
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0,    # no call
# ]

# assert play_move(EXAMPLE_STATE_FIVE) == 14 # three 3s



# EXAMPLE_STATE_SIX = [
#     0, # not opponent's turn
#     1, # our turn
#     1, 0, 0, 0, 0, 0, # die_1 is 1
#     1, 0, 0, 0, 0, 0, # die_2 is 1
#     0, 0, 1, 0, 0, 0, # die_3 is 3 
#     0, 0, 1, 0, 0, 0, # die_4 is 3 
#     0, 0, 0, 0, 1, 0, # die_5 is 5 
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 1, 0, 0, 0, 0, # three 2s bid
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0,
#     0,    # no call
# ]

# assert play_move(EXAMPLE_STATE_SIX) == 60 # CALL


# move = play_move(EXAMPLE_STATE_SIX)
# print(move, interpret_move(move))
