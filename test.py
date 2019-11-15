from src.NN import NetWrapper
from src.games.Tictactoe import Tictactoe
from src.Player import * 
from src.MCTS import MCTS
import yaml

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

game = Tictactoe(**config['GAME'])
nn = NetWrapper(game, **config['NN'])
nn.load_model("models/old_model.pt")
nn1 = NetWrapper(game, **config['NN'])
nn1.load_model()
mcts = MCTS(**config['MCTS'])

player_vs_player(game, p1 =  AlphaZeroPlayer(nn1, mcts), p2 = AlphaZeroPlayer(nn, mcts), n_games = 50, treshold = 0.8,  print_b = True)
