import yaml
import optuna
from src.AlphaZeroTrainer import AlphaZeroTrainer as az 
from src.NN import NetWrapper
from src.games.Tictactoe import Tictactoe
from src.Player import * 
from src.MCTS import MCTS

#{'lr': 0.012486799182525229, 'wd': 0.015010724465999522}.

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)
game = Tictactoe(**config['GAME'])
mcts = MCTS(**config['MCTS'])
alphat = az(nn, game, mcts, **config['AZ'])
loss = alphat.train()
