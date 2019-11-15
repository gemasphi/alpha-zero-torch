from src.AlphaZeroTrainer import AlphaZeroTrainer as az 
from src.NN import NetWrapper as nn
from src.games.Tictactoe import Tictactoe
from src.Player import * 
from src.MCTS import MCTS
import yaml

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

game = Tictactoe(**config['GAME'])
nn = nn(game, **config['NN'])
mcts = MCTS(**config['MCTS'])
alphat = az(nn, game, mcts, **config['AZ'])
alphat.train()