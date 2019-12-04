import math
import numpy as np
import torch
import pickle
import collections
from multiprocessing.dummy import Pool as ThreadPool

#TODO add asserts

class MCTS(object):
	def __init__(self, **params):
		super(MCTS, self).__init__()
		self.params = params
		self.cpuct = params['cpuct']
		self.n_simulations = params['n_simulations']
		self.dirichlet_alpha = params['dirichlet_alpha']

	def new_mcts(self):
		return MCTS(**self.params)

	def simulate(self, game, nn, temp = 1):
		root = GameState(game, None, DummyState())
		for i in range(self.n_simulations):
			leaf = root.select(self.cpuct)
			
			if leaf.winner != None:
				leaf.backup(leaf.winner)
				continue

			v = leaf.expand(nn, self.dirichlet_alpha)
			leaf.backup(v)
	
		return root.get_search_policy(temp)

	def paralell_simulate(self, game, nn, temp = 1):
		root = GameState(game, None, DummyState())
		leafies = []
		eval_size = 8
		simulations_to_run = self.n_simulations // eval_size 
		root = GameState(game, None, DummyState())
		pool = ThreadPool(eval_size)
	
		#we do more than the n_simulations currently
		for _ in range(simulations_to_run + 1):
			leafies = pool.map(self.virtual_loss_search, [root for i in range(eval_size)])
			leafies = list(filter(None, leafies)) #hacky way of dealing with end states
			if len(leafies) == 0:
				break

			v, p = nn.predict([l.game.get_canonical_board() for l in leafies]) #todo:check values according to player or something 
			pool.map(self.virtual_loss_backup, [(leafies[i], p[i], v[i]) for i in range(len(leafies))])

		return root.get_search_policy(temp)

	def virtual_loss_search(self, root):
		leaf = root.select(self.cpuct)
		if leaf.winner != None:
			leaf.backup(leaf.winner)
			return None

		leaf.add_virtual_loss()

		return leaf

	def virtual_loss_backup(self, args):
		leaf, p, v = args
		leaf.remove_virtual_loss()
		leaf.expand_s(p)
		leaf.backup(v)

class DummyState(object):
    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)


class GameState():
	def __init__(self, game, action, parent):
		super(GameState, self).__init__()
		self.parent = parent
		self.game = game
		self.winner = self.game.check_winner() 
		self.action = action
		self.is_expanded = False

		self.child_N = np.zeros([game.get_action_size()], dtype=np.float32)
		self.child_W = np.zeros([game.get_action_size()], dtype=np.float32)
		self.child_P = np.zeros([game.get_action_size()], dtype=np.float32)
		self.children = {}

	def add_virtual_loss(self):
		current = self
		while current.parent is not None:
			current.update_W(1)
			current = current.parent

	def remove_virtual_loss(self):
		current = self
		while current.parent is not None:
			current.update_W(-1)
			current = current.parent

	def select(self, cpuct):
		current = self
		while current.is_expanded and current.winner == None:
			action = np.argmax(current.child_Q() + current.child_U(cpuct))
			current = current.play(action)

		return current

	def play(self, action):
		if action not in self.children:
			new_game = self.game.copy_game()
			new_game.play(action)
			self.children[action] = GameState(new_game, action, parent = self) 

		return self.children[action]

	def backup(self, v):
		current = self
		while current.parent is not None:
			current.inc_N()
			current.update_W(v)
			current = current.parent

	def update_W(self,v):
		self.parent.child_W[self.action] -= v

	def inc_N(self):
		self.parent.child_N[self.action] += 1

	def get_N(self):
		return self.parent.child_N[self.action]

	def get_W(self):
		return self.parent.child_W[self.action]

	def get_P(self):
		return self.parent.child_P[self.action]

	def child_Q(self):
		return  self.child_W / (1 + self.child_N)

	def child_U(self, cpuct):
		return cpuct*self.child_P*(math.sqrt(self.get_N())/(1 +self.child_N))

	def expand_s(self, p):
		self.is_expanded = True

		p = p.flatten()
		poss = self.game.get_possible_actions()
		self.child_P = self.get_valid_actions(p, poss, self.game)
		#if self.parent == None:
		#	d = np.random.dirichlet(np.ones(len(p))*dirichlet_alpha)
		#	p = 0.75*p + 0.25*d

		
	def expand(self, nn, dirichlet_alpha):
		self.is_expanded = True

		v, p = nn.predict(self.game.get_canonical_board())
		p = p.flatten()
		poss = self.game.get_possible_actions()
		
		#if self.parent == None:
		#	d = np.random.dirichlet(np.ones(len(p))*dirichlet_alpha)
		#	p = 0.75*p + 0.25*d

		self.child_P = self.get_valid_actions(p, poss, self.game)
		
		return v 

	def get_valid_actions(self, pred, poss, game):
		valid_actions = (pred * poss)

		if not np.any(valid_actions): 			#case where there are no valid actions, we use possible actions
			valid_actions = poss
		return valid_actions/sum(valid_actions)

	def get_search_policy(self, temp):
		count = self.child_N**(1/temp)
		return count/np.sum(count)
