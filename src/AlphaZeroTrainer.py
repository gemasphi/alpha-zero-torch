import numpy as np
from .utils.plot import unique_positions_vis
from multiprocessing.dummy import Pool as ThreadPool

class AlphaZeroTrainer(object):
	def __init__(self, NN, game, mcts, **params):
		super(AlphaZeroTrainer, self).__init__()
		self.nn_wrapper = NN
		self.game = game
		self.mcts = mcts
		self.queue_len = params['queue_len']
		self.n_games = params['n_games']
		self.eps = params['eps']
		self.temp = params['temp']
		self.replay_buffer = ReplayBuffer(self.queue_len)

	def train(self, lr, wd):
		pool = ThreadPool(4)

		for i in range(self.eps):
			#for j in range(self.n_games):
			#	self.play_game(j)

			pool.map(self.play_game, range(self.n_games))

			#print(self.replay_buffer.buffer)
			#for j in range(self.n_games):
			#	print("One game played {}/{}".format(j, self.n_games))

			loss = self.nn_wrapper.train(self.replay_buffer, lr = lr ,wd = wd)
			self.nn_wrapper.save_model()
			print("One self play ep: {}/{}, avg loss: {}".format(i,self.eps, loss))

		return loss

	def play_game(self, n_game):
		winner = None
		game = self.game.new_game()
		mcts = self.mcts.new_mcts()
		game_step = 0
		temp = self.temp['before']
		print(f'plaing game number{n_game}')
		while winner == None:
			if game_step < self.temp['treshold']:
				temp = self.temp['after']
			
			action_probs = mcts.paralell_simulate(game, self.nn_wrapper, temp)
			action = np.random.choice(len(action_probs),p = action_probs) #TODO: should this be uniform or not?
			game.play(action)
				
			winner = game.check_winner()
			game_step += 1
		
		self.replay_buffer.save_game(game)

class ReplayBuffer(object):
	def __init__(self, window_size):
		super(ReplayBuffer, self).__init__()
		self.buffer = []
		self.window_size = window_size

	def save_game(self, game):
	    if len(self.buffer) > self.window_size:
	      self.buffer.pop(0)
	    self.buffer.append(game)

	def sample_batch(self, batch_size):
	    n_positions = self.get_total_positions()

	    games = np.random.choice(
	        self.buffer,
	        size= batch_size)

	    game_pos = [(g, np.random.randint(len(g.history))) for g in games]
	    pos = np.array([[g.make_input(i), *g.make_target(i)] for (g, i) in game_pos])
	    return list(pos[:,0]), list(pos[:,1]), list(pos[:,2])

	def get_total_positions(self):
		return float(sum(len(g.history) for g in self.buffer))
