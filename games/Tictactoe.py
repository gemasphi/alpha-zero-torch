from .Game import Game
import numpy as np

class Tictactoe(Game):
	def __init__(self,  player = 1, **params):
		self.params = params
		self.board_size = params['board_size']
		self.board = np.zeros(params['board_size'])  
		self.in_row = params['in_row']
		self.player = player

	def set_board(self, board):
		self.board = board

	def reset(self):
		self.player = 1
		self.board = np.zeros(self.board_size)

	def get_action_size(self):
		return self.board_size[0]*self.board_size[1]

	def get_board_dimensions(self):
		return self.board_size

	def get_input_planes(self):
		return 1

	def get_player(self):
		return player

	def get_output_planes(self):
		return 1

	def get_possible_actions(self):
		p = []
		for row in self.board:
			for el in row:
				if el == 0:
					p.append(1)
				else:
					p.append(0)

		return p

	def get_possible_actions_index(self):
		return np.argwhere(self.board == 0)

	def get_player(self):
		return self.player

	def play(self, action):
		x, y = action//self.board_size[0], action%self.board_size[1]
		assert self.board[x][y] == 0, "Invalid action, this shouldn't happen"

		self.board[x][y] = self.player
		self.player = self.player*-1

	def check_winner(self):
		if len(np.argwhere(self.board == 0)) == 0:
			return 0

		for p in [-1,1]:
			player_positions = (self.board == p)
			if self.find_win(player_positions):
				return p*self.player			

		return None

	def find_win(self, player_positions):
		for i in range(self.board_size[0] - self.in_row + 1):
			for j in range(self.board_size[1] - self.in_row + 1):
				if self.is_win(player_positions[i : i + self.in_row, j : j + self.in_row]):
					return True
		return False

	def is_win(self, small_board):
		vertical = np.all(small_board, axis = 0)
		horizontal = np.all(small_board, axis = 1)
		diagonals = np.all(small_board.diagonal()) \
					or np.all(np.fliplr(small_board).diagonal())

		return np.any(vertical) or np.any(horizontal) or diagonals

	def print_board(self):
		print(np.array2string(self.board, formatter={'float': lambda x: self.pos2string(x)}))

	def pos2string(self, x):
		if x == 1:
			return 'X'
		elif x == -1:
			return 'O'
		else:
			return '-'

	def get_board(self):
		return self.board

	def get_input_representation(self):
		return self.board

	def hashed_board(self):
		return hash(str(self.board))

	def get_canonical_board(self):
		return self.board*self.player

	def copy_game(self):
		new_game = Tictactoe(player = self.get_player(), **self.params)
		new_game.set_board(np.copy(self.get_board()))
		return new_game