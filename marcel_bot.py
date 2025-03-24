from merlob_bot import MerlobBot

class MarcelBot:
	def __init__(self, variable = 0):
		self.variable = variable

	def play_move(self, state):
		merlob_bot = MerlobBot()
		return merlob_bot.play_move(state)
