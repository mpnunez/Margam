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