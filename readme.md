# Connect4 AI

Train a bot to play Connect4

Uses Deep Q-learning from Chapter 6 of Deep Reinforcement Learning Hands-On [1]. Also uses policy gradient from Chapter 9.


[1] Lapan, Maxim. Deep Reinforcement Learning Hands-On: Apply modern RL methods, with deep Q-networks, value iteration, policy gradients, TRPO, AlphaGo Zero and more. Packt Publishing Ltd, 2018.

Developer
- Marcel Nunez (marcelpnunez@gmail.com)

## Setup

I used python 3.8.0

```
python3.8 -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.in
```

To launch an monitor a training:

Step 1: Start the training. For example, to train a DQN agent on Tic-Tac-Toe

```
python main.py train -a dqn -g tic_tac_toe
```

Step 2: Launch tensorboard

```
tensorboard serve --logdir runs
```

Step 3: Open [tensorboard](http://localhost:6006) from a web browser.


## Results

Win rates vs. 1-step minimax opponent.

| Agent | Tic-tac-toe | Connect 4 |
| ------|-------------|-----------|
| PG    | 63%         |    63%    |
| DQN   | 87%         |    89%    |

Training for each agent took 1-2 hours on my laptop with no GPU.
