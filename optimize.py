import yaml
import optuna
from src.AlphaZeroTrainer import AlphaZeroTrainer as az 
from src.NN import NetWrapper
from src.games.Tictactoe import Tictactoe
from src.Player import * 
from src.MCTS import MCTS

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

game = Tictactoe(**config['GAME'])
mcts = MCTS(**config['MCTS'])

def objective(trial):
	nn = NetWrapper(game, **config['NN'])
	lr = trial.suggest_loguniform('lr', 0.01, 0.5)
	wd = trial.suggest_loguniform('wd', 0.01, 0.5)
	alphat = az(nn, game, mcts, **config['AZ'])
	loss = alphat.train(lr, wd)
	return loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials = 100)
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial
print('  Value: ', trial.value)
