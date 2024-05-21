import numpy as np


class Connect4Exception(Exception):
    pass

def get_training_and_viewing_state(game,state):
    """
    For tic-tac-toe and Connect4, convert the
    open_spiel tensor to something trainable
    and viewable from current player POV

    state_np_for_conv: state as numpy array that can be
        fed to a convolutional neural network
        Dimensions 0 and 1 are rows and columns
        Dimension 2 are different players
    human_view_state: 2D numpy array with vacancies
        and different player tokens as different integers
    """

    state_as_tensor = state.observation_tensor()
    tensor_shape = game.observation_tensor_shape()
    state_np = np.reshape(np.asarray(state_as_tensor), tensor_shape)
    
    # Remove last element of 1st dimension showing empty spaces
    state_np = state_np[1::-1,:,:]

    # Move players axis last to be the channels for conv net
    state_np_for_cov = np.moveaxis(state_np, 0, -1)
    
    # view as 1 2D matrix with the last row being first
    human_view_state = state_np[0,::-1,:]+2*state_np[1,::-1,:]

    return state_np_for_cov, human_view_state

def generate_transitions(agent, opponents):
    """
    Infinitely yield transitions by playing
    game episodes
    """

    for i, _ in enumerate(iter(bool, True)):

        opponent_ind = (i//2)%len(opponents)    # Play each opponent twice in a row
        opponent = opponents[opponent_ind]
        agent_position = i%2
        opponent_position = (agent_position+1)%2
        
        g = TicTacToe(nrows=NROWS,ncols=NCOLS,nconnectwins=NCONNECT)
        g.players = [None,None]
        g.players[agent_position] = agent        # Alternate being player 1/2
        g.players[opponent_position] = opponent   
        
        
        winner, records = g.play_game()
        agent_records = records[agent_position::len(g.players)]

        agent_records_td = []
        for i, tr in enumerate(agent_records):
            td_tsn = Transition(
                state = tr.state,
                selected_move = tr.selected_move,
                reward = tr.reward,
            )
            
            for j in range(i+1, min( len(agent_records), i+N_TD) ):
                td_tsn.reward += agent_records[j].reward * DISCOUNT_RATE ** (j-i)

            if i + N_TD < len(agent_records):
                td_tsn.next_state = agent_records[i+N_TD].state

            agent_records_td.append(td_tsn)

        for move_record in agent_records_td:
            yield move_record, opponent