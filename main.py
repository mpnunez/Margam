import click
import pyspiel
from c4lib.player import HumanPlayer, MiniMax
import numpy as np
import random
import yaml


@click.group()
def main():
    pass


@main.command()
@click.option(
    "-g",
    "--game-type",
    type=click.Choice(list(pyspiel.registered_names()), case_sensitive=False),
    default="connect_four",
    show_default=True,
    help="game type",
)
@click.option(
    "-o",
    "--opponent",
    type=click.Choice(["human", "minimax", "dqn", "pg"], case_sensitive=False),
    default="minimax",
    show_default=True,
    help="opponent type",
)
@click.option(
    "-d", "--depth", type=int, default=2, show_default=True, help="Depth for minimax"
)
@click.option(
    "-m",
    "--model",
    type=str,
    default=None,
    help="Model file to load for AI player",
)
@click.option("--second", is_flag=True, default=False, help="Play as second player")
def play(game_type, opponent, depth, model, second):

    # Intialize players
    human = HumanPlayer(name="Marcel")

    opponent = opponent.lower()
    if opponent == "minimax":
        opponent = MiniMax(name="Maximus", max_depth=depth)
    elif opponent == "human":
        opponent = HumanPlayer("Opponent")
    elif opponent == "pg":
        from c4lib.train_pg import PolicyPlayer
        from keras.models import load_model

        opponent = PolicyPlayer(name="PG")
        if game_type == "liars_dice":
            from c4lib.train_pg import initialize_model
            opponent.model = initialize_model(
                game_type, {"ACTOR_CRITIC":False}, show_model=True)
        else:
            opponent.model = load_model(model)
        opponent.model.summary()
    elif opponent == "dqn":
        from c4lib.train_dqn import DQNPlayer
        from keras.models import load_model

        opponent = DQNPlayer(name="DQN")
        opponent.model = load_model(model)
        opponent.model.summary()

    players = [opponent, human] if second else [human, opponent]

    if game_type == "liars_dice":
        game = pyspiel.load_game(game_type,{"numdice":5})
        # You can pass a dictionary as an optional second argument
        # to load_game to pass game parameters. For liars poker the
        # default number of die is 1.
    else:
        game = pyspiel.load_game(game_type)
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            # Sample a chance event outcome.
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
        else:
            # If the player action is legal, do it. Otherwise, do random
            current_player = players[state.current_player()]
            action = current_player.get_move(game, state)
            if action not in state.legal_actions():
                action = random.choice(state.legal_actions())
            state.apply_action(action)

    print(state.returns())
    winner = np.argmax(state.returns())
    print(f"{players[winner].name} won!")


@main.command()
@click.option(
    "-g",
    "--game-type",
    type=click.Choice(list(pyspiel.registered_names()), case_sensitive=False),
    default="tic_tac_toe",
    show_default=True,
    help="game type",
)
@click.option(
    "-a",
    "--algorithm",
    type=click.Choice(["dqn", "pg"], case_sensitive=False),
    default="dqn",
    show_default=True,
    help="Reinforcement learning algorithm",
)
@click.option(
    "-h",
    "--hyperparameter-file",
    type=str,
    default="hyperparams.yaml",
    show_default=True,
    help="YAML file with hyperparameter values",
)
def train(game_type, algorithm, hyperparameter_file):

    with open(hyperparameter_file, "r") as f:
        hp = yaml.safe_load(f)

    if algorithm == "dqn":
        from c4lib.train_dqn import train_dqn
        train_dqn(game_type, hp[game_type])
    elif algorithm == "pg":
        from c4lib.train_pg import train_pg
        train_pg(game_type, hp[game_type])


if __name__ == "__main__":
    main()
