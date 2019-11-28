from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def unique_positions_vis(pos, game, name = "unique_pos_vis", show = False):
	pos_moves_count = moves_count(pos, game)
	#pos_count = Counter(str(t[0]) for t in pos)

	x, y = [], []
	for i in range(1, game.get_action_size() + 1):
		x.append(i)
		if i in pos_moves_count:
			y.append(len(set(pos_moves_count[i])))
		else:
			y.append(0)

	print(x)
	print(y)
	"""
	fig, ax = plt.subplots()
	plt.bar(x, y)

	if show:
		plt.show()
	else:
		fig.savefig('plots/unique_pos_vis_{}.png'.format(name))
	"""
def moves_count(pos, game):
	res = {}
	for p in pos:
		p = p[0]
		game.set_board(p)
		n_moves_played = game.get_action_size() - np.count_nonzero(game.get_possible_actions())

		if n_moves_played in res:
			res[n_moves_played].append(str(p))
		else:
			res[n_moves_played] = [str(p)]

	return res
