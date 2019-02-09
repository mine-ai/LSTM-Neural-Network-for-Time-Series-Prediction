__copyright__ = "Jakob Aungiers 2018 MIT"
import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_loader import DataLoader
from core.model import Model
import logging

log_fmt = '%(asctime)s %(levelname)s %(message)s %(name)s.%(funcName)s:%(lineno)d'
logging.basicConfig(
    level=logging.DEBUG,
    format=log_fmt,
    filemode='a',)
log = logging.getLogger()


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


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


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', 'sp500.csv'),
        0.85,
        [
            "Close",
            "Volume"
        ],
    )

    model = Model()
    SEQUENCE_LENGTH = 50
    BATCH_SIZE = 32
    NORMALISE = True
    EPOCHS = 3
    SAVE_DIR = 'saved_models'

    x, y = data.get_train_data(
        seq_len=SEQUENCE_LENGTH,
        normalise=NORMALISE
    )
    # out-of memory generative training
    steps_per_epoch = math.ceil((data.len_train - SEQUENCE_LENGTH) / BATCH_SIZE)
    model.train_gen(
        data_gen=data.gen_train_batch(
            seq_len=SEQUENCE_LENGTH,
            batch_size=BATCH_SIZE,
            normalise=NORMALISE
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        steps_per_epoch=steps_per_epoch,
        save_dir=SAVE_DIR,
    )

    x_test, y_test = data.get_test_data(
        seq_len=SEQUENCE_LENGTH,
        normalise=NORMALISE,
    )

    predictions = model.predict_sequences_multiple(x_test, SEQUENCE_LENGTH, SEQUENCE_LENGTH)
    # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    # predictions = model.predict_point_by_point(x_test)

    plot_results_multiple(predictions, y_test, SEQUENCE_LENGTH)
    # plot_results(predictions, y_test)


if __name__ == '__main__':
    main()
