# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:55:45 2023

@author: tiank
"""

import pytest
from connect4lib.game import Game

@pytest.mark.parametrize("player_moves,is_there_a_win", 
                         [
                             ([], False),
                             ([(0,0),(1,1),(2,2),(3,3)], True),
                             ([(2,1),(2,1),(2,1),(2,0)], True),
                            ([(0,0),(1,0),(0,1),(1,2),(2,4),(3,4)], True),
                            ([(6,6),(5,6),(6,5),(5,4),(4,2),(3,2)], True),
                            ([(0,0),(1,1),(2,2),(4,3)], False),
                        ])
def test_detect_win(player_moves,is_there_a_win):
    g = Game()
    for player1_move, player2_move in player_moves:
        g.drop_in_slot(0,player1_move)
        g.drop_in_slot(1,player2_move)
    assert g.check_win() == is_there_a_win