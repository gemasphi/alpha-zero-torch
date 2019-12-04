import math
import numpy as np
import torch
import pickle

#TODO add asserts

class MCTS(object):
	def __init__(self, **params):
		super(MCTS, self).__init__()
		self.game_states = {}
		self.params = params
		self.cpuct = params['cpuct']
		self.n_simulations = params['n_simulations']
		self.dirichlet_alpha = params['dirichlet_alpha']

	def reset(self):  #discards tree TODO: there's no need to completly discard the tree
		self.game_states = {}

	def new_mcts(self):
		return MCTS(**self.params)

	def simulate(self, game, nn, temp = 1):
		self.root_node = str(game.get_canonical_board())
		self.nn_wrapper = nn

		for i in range(self.n_simulations):
			self.search(game)

		p = self.calc_policy(self.game_states[self.root_node], temp) 
		game.add_policy(p)
		return p

	def calc_policy(self, game_state, temp):
		count = game_state.get_edge_count()
		count_t = count**(1/temp)

		return count_t/sum(count_t)

	def search(self, game):
		s = str(game.get_canonical_board())
		winner = game.check_winner() 

		if winner != None:
			return -winner

		if not self.game_expanded(s):
			return -self.expand(s, game)
		else:
			game_state = self.game_states[s]
			best_action = game_state.select(self.cpuct)
			n_game = game_state.play(best_action)
			v = self.search(n_game) 
			game_state.backup(best_action, v)

			return -v 

	def expand(self, s, game):
		v, p = self.nn_wrapper.predict(game.get_canonical_board())
		p = p.flatten()
		poss = game.get_possible_actions()
		
		if self.root_node == s:
			d = np.random.dirichlet(np.ones(len(p))*self.dirichlet_alpha)
			p = 0.75*p + 0.25*d

		valid_actions = self.get_valid_actions(p, poss, game)
		self.game_states[s] = GameState(game, policy = valid_actions)

		return v 

	def get_valid_actions(self, pred, poss, game):
		valid_actions = (pred * poss)

		if not np.any(valid_actions): 			#case where there are no valid actions, we use possible actions
			valid_actions = poss

		return valid_actions/sum(valid_actions)

	def game_expanded(self, s):
		return s in self.game_states


class GameState():
	def __init__(self, game, policy = 0):
		super(GameState, self).__init__()
		self.game = game
		self.N = 0
		self.policy = policy       
		self.edges = {a: Edge() for a in range(len(policy)) if policy[a] != 0 }

	def select(self, cpuct):
		max_u = -10
		for a, edge in self.edges.items(): 
			u = edge.get_Q() + cpuct*self.policy[a]*math.sqrt(self.N)/(1+edge.get_N())			

			if u > max_u: 
				max_u = u
				best_action = a

		return best_action

	def play(self, action):
		new_game = self.game.copy_game()

		new_game.play(action)
		return new_game 
	
	def backup(self, a, v):
		self.edges[a].backup(v)
		self.N += 1

	def get_N(self):
		return self.N

	def valid_actions(self):
		return self.policy

	def get_game(self):
		return self.game

	def get_edge_count(self):
		count = np.zeros(len(self.policy))

		for a, edge in self.edges.items():
			count[a] =  edge.get_N()
		return count

class Edge():
	def __init__(self, Q = 0, N = 0, W = 0):
		super(Edge, self).__init__()
		self.Q = Q
		self.N = N
		self.W = W  

	def backup(self, v):
		self.W += v
		self.N += 1
		self.Q = self.W/self.N
		
		return self.Q
	
	def get_N(self):
		return self.N

	def get_Q(self):
		return self.Q