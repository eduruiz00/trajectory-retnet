import time

class Timer:

	def __init__(self):
		self._start = time.time()
		self._last = self._start

	def __call__(self, reset=True, flag=False):
		now = time.time()
		diff = now - self._start
		if flag:
			diff_last = now - self._last
			self._last = now
			return diff, diff_last
		if reset:
			self._start = now
		return diff