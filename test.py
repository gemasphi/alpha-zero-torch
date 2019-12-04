from src.NN import NetWrapper
from src.games.Tictactoe import Tictactoe
from src.Player import * 
from src.MCTS import MCTS
import yaml

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

game = Tictactoe(**config['GAME'])
"""
nn = NetWrapper(game, **config['NN'])
nn.load_model("models/the_bestest_of_models.pt")
"""
nn1 = NetWrapper(game, **config['NN'])
nn1.load_model()

mcts = MCTS(**config['MCTS'])

play_game(game, p1 =  AlphaZeroPlayer(nn1, mcts), p2 =  HumanPlayer(), print_b = True)
#player_vs_player(game, p1 =  AlphaZeroPlayer(nn, mcts),  p2 =  AlphaZeroPlayer(nn1, mcts), n_games = 100, treshold = 0.5, print_b = False)
