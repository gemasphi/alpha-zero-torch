import numpy as np
from collections import deque
from Player import *
from NN import NetWrapper as nn

class AlphaZeroTrainer(object):
	def __init__(self, NN, game, mcts, **params):
		super(AlphaZeroTrainer, self).__init__()
		self.nn_wrapper = NN
		self.game = game
		self.mcts = mcts
		self.queue_len = params['queue_len']
		self.n_games = params['n_games']
		self.eps = params['eps']

	def train(self):
		train_data = deque([], maxlen = self.queue_len)

		for i in range(self.eps):
			print("One self play ep: {}/{}".format(i,self.eps))
			train_data += self.generate_data()
			self.nn_wrapper.train(train_data)
			self.nn_wrapper.save_model()
			
	def generate_data(self):
		train_examples = deque([])

		for i in range(self.n_games):
			print("Games played: {}/{}".format(i,self.n_games))
			generated = self.self_play()
			train_examples += generated 

		return train_examples

	def self_play(self):
		examples = []
		winner = None
		self.game.reset()
		while winner == None:
			action_probs = self.mcts.simulate(self.game, self.nn_wrapper)
			action = np.random.choice(len(action_probs),p = action_probs) #TODO: should this be uniform or not?
				
			examples.append([np.copy(self.game.get_board()), action_probs, self.game.get_player()])
			self.game.play(action)
			winner = self.game.check_winner()

		examples.append([np.copy(self.game.get_board()), action_probs, self.game.get_player()])

		return [(ex[0]*ex[2], ex[1], winner*self.game.get_player()*ex[2]) for ex in examples[1:]]