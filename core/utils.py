import datetime as dt
import logging

log = logging.getLogger()

class Timer():

	def __init__(self):
		self.start_dt = None

	def __enter__(self):
		self.start_dt = dt.datetime.now()

	def __exit__(self, exc_type, exc_val, exc_tb):
		end_dt = dt.datetime.now()
		log.info('Time taken: %s' % (end_dt - self.start_dt))

	def start(self):
		self.start_dt = dt.datetime.now()

	def stop(self):
		end_dt = dt.datetime.now()
		print('Time taken: %s' % (end_dt - self.start_dt))
