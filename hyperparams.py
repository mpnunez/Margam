# Game hyperparameters

GAME_TYPE = "TicTacToe"
#GAME_TYPE = "Connect4"

if GAME_TYPE == "Connect4":
    NROWS = 6
    NCOLS = 7
    NPLAYERS = 2
    NCONNECT = 4
    NOUTPUTS = NCOLS

    # Learning
    DISCOUNT_RATE = 0.97
    LEARNING_RATE = 1e-3

    # Recording progress
    REWARD_BUFFER_SIZE = 1_000
    RECORD_HISTOGRAMS = 1_000
    SAVE_MODEL_ABS_THRESHOLD = 0.20
    SAVE_MODEL_REL_THRESHOLD = 0.01

    # DQN
    BATCH_SIZE = 32             
    REPLAY_SIZE = 10_000
    SYNC_TARGET_NETWORK = 1_000
    REPLAY_START_SIZE = 10_000
    EPSILON_DECAY_LAST_FRAME = 5e4
    EPSILON_START = 1.0
    EPSILON_FINAL = 0.02

    # Policy gradient
    BATCH_N_EPISODES = 4
    ENTROPY_BETA = 0.1

elif GAME_TYPE == "TicTacToe":
    NROWS = 3
    NCOLS = 3
    NPLAYERS = 2
    NCONNECT = 3
    NOUTPUTS = NROWS*NCOLS

    # Learning
    DISCOUNT_RATE = 0.97
    LEARNING_RATE = 1e-4

    # Recording progress
    REWARD_BUFFER_SIZE = 1_000
    RECORD_HISTOGRAMS = 1_000
    SAVE_MODEL_ABS_THRESHOLD = -0.6
    SAVE_MODEL_REL_THRESHOLD = 0.01

    # DQN
    BATCH_SIZE = 32             
    REPLAY_SIZE = 1_000
    SYNC_TARGET_NETWORK = 1_00
    REPLAY_START_SIZE = 1_000
    EPSILON_DECAY_LAST_FRAME = 3e4
    EPSILON_START = 1.0
    EPSILON_FINAL = 0.02

    # Policy gradient
    BATCH_N_EPISODES = 4
    ENTROPY_BETA = 0.1