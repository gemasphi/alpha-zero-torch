from .MCTS import MCTS
import numpy as np
from math import inf as infinity

def play_game(game, p1, p2, print_b = False):
	game.reset()
	winner = None
	current_player = p1 

	while winner == None:
		action = current_player.get_action(game)
		game.play(action)
		winner = game.check_winner()
		current_player = p2 if current_player == p1 else p1
		
		if print_b:
			game.print_board()

	return winner*game.get_player()

def player_vs_player(game, p1, p2, n_games = 10, treshold = 0.5, print_b = False): 
	draws = 0
	wins_p1 = 0
	for i in range(n_games):
		print("Game: {}/{}".format(i,n_games))
		winner = play_game(game = game, p1 = p1, p2 = p2, print_b = print_b) #todo: we should probabily vary who plays first, some games are biased torwards first players
		if winner == 0:
			draws += 1
		elif winner == 1:
			wins_p1 +=1

	print("Player 1 won:{}%, lost:{}%, drew:{}%".format(wins_p1, n_games - wins_p1 - draws, draws))

	winner_model = p1 if wins_p1/n_games > treshold else p2

	return winner


class Player(object):
	def __init__(self):
		super(Player, self).__init__()

	def get_action(self, game):
		pass

class RandomPlayer(Player):
	def __init__(self):
		super(RandomPlayer, self).__init__()

	def get_action(self, game):
		possible_actions = np.nonzero(game.get_possible_actions())[0]

		return np.random.choice(possible_actions)

class AlphaZeroPlayer(Player):
	def __init__(self, nn, mcts):
		super(AlphaZeroPlayer, self).__init__()
		self.mcts = mcts
		self.nn = nn

	def get_action(self, game):
		action_probs = self.mcts.simulate(game, self.nn)
		action = np.argmax(action_probs)
		print(action_probs)
		return action

	def get_nn(self):
		return self.nn

class HumanPlayer(Player):
	def __init__(self):
		super(HumanPlayer, self).__init__()

	def get_action(self, game):
		possible_actions = np.nonzero(game.get_possible_actions())[0]
		
		action = input ("Enter an action:")
		action = int(action) - 1
		
		while action not in possible_actions:
			action = input ("Invalid action. Enter an action:")
			action = int(action) - 1

		return action

class NNPlayer(Player):
	def __init__(self, nn):
		super(NNPlayer, self).__init__()
		self.nn = nn
		
	def get_action(self, game):
		v, action_probs = self.nn.predict(game.get_canonical_board())
		action_probs = action_probs.flatten()
		poss = game.get_possible_actions()
		action_probs = action_probs * poss
		print(action_probs)
		print(v)
		action = np.argmax(action_probs)
		return action


class MinimaxPlayer(Player):
	def __init(self):
		super(MinimaxPlayer, self).__init__()
	def get_action(self, game):
		self.pos_explored = 0
		return self.minimax(9, game, game.get_player())

	def minimax(self, depth, game, initial_player):
		self.pos_explored += 1
		if self.pos_explored % 1000 == 0:
			print(self.pos_explored)
			print(depth)

		if game.get_player() == initial_player:
			best = [-1, -infinity]
		else:
			best = [-1, +infinity]

		winner = game.check_winner()

		if depth == 0 or winner != None:
			return [-1 , winner]

		poss_actions_index = game.get_possible_actions_index()
		for action in poss_actions_index:
			n_game = game.copy_game()
			n_game.play(action)
			score = self.minimax(depth - 1, n_game, initial_player)
			score[0] = action

			if game.get_player() == initial_player:
				if score[1] > best[1]:
					best = score  # max value
			else:
				if score[1] < best[1]:
					best = score  # min value

		return best


