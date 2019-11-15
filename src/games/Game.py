class Game(object):
	def __init__(self, board_size, board = None, player = 1):
		super(Game, self).__init__()

	def get_action_size(self):
		pass

	def get_board_dimensions(self):
		pass

	def get_input_planes(self):
		pass

	def get_output_planes(self):
		pass

	def get_possible_actions(self, board):
		pass

	def play(self, action):
		pass

	def get_canonical_board(self):
		pass

	def copy_game(self):
		pass

	def reset(self):
		pass