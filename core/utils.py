import datetime as dt
import logging

from matplotlib import pyplot as plt

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


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
