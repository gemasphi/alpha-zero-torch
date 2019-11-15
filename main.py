import yaml
from AlphaZeroTrainer import AlphaZeroTrainer as az 
from NN import NetWrapper as nn
from games.Tictactoe import Tictactoe
from Player import * 
from MCTS import MCTS

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

game = Tictactoe(**config['GAME'])
nn = nn(game, **config['NN'])
#nn.load_model()
mcts = MCTS(**config['MCTS'])
alphat = az(nn, game, mcts, **config['AZ'])
alphat.train()

#game.set_board(np.array([[1,1,1],[0,1,0],[0,0,1]]))
#print(game.check_winner())
#game.print_board()
#player_vs_player(game, p1 =  RandomPlayer(), p2 =AlphaZeroPlayer(nn, mcts), n_games = 50, treshold = 0.8,  print_b = True)

#play_game(game, p1 = AlphaZeroPlayer(nn, mcts), p2 = HumanPlayer(), print_b = True)