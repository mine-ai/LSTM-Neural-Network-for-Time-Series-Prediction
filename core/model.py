import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from core.utils import Timer
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time

import logging
log = logging.getLogger()


class Model():
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        t0 = time.clock()

        self.model = Sequential()

        input_timesteps = 49
        units = 100
        input_dim = 2
        self.model.add(LSTM(units, input_shape=(input_timesteps, input_dim), return_sequences=True))

        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units, input_shape=(None, None), return_sequences=True))
        self.model.add(LSTM(units, input_shape=(None, None), return_sequences=False))

        self.model.add(Dropout(0.2))

        self.model.add(Dense(1, activation='linear'))

        self.model.compile(loss='mse', optimizer='adam')

        t1 = time.clock()

        log.info(f'[Model] Model Compiled, used time: {t1 - t0} seconds')

    def load(self, filepath):
        log.info(f'[Model] Loading model from file {filepath}')
        self.model = load_model(filepath)

    # def train(self, x, y, epochs, batch_size, save_dir):
    #     with Timer() as timer:
    #         log.info('[Model] Training Started')
    #         log.info('[Model] %s epochs, %s batch size' % (epochs, batch_size))
    #
    #         save_fname = os.path.join(save_dir, '%s-e%s.h5' % (
    #         dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
    #         callbacks = [
    #             EarlyStopping(monitor='val_loss', patience=2),
    #             ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
    #         ]
    #         self.model.fit(
    #             x,
    #             y,
    #             epochs=epochs,
    #             batch_size=batch_size,
    #             callbacks=callbacks
    #         )
    #         self.model.save(save_fname)
    #
    #         log.info(f'[Model] Training Completed. Model saved to: {save_fname}')

    def train_gen(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        with Timer() as timer:
            log.info('[Model] Training Started')
            log.info(f'[Model] {epochs} epochs, {batch_size} batch size, {steps_per_epoch} batches per epoch')

            save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
            callbacks = [
                ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
            ]

            self.model.fit_generator(
                data_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                callbacks=callbacks,
                workers=1
            )

            log.info(f'[Model] Training Completed. Model saved as {save_fname}')

    def predict_point_by_point(self, data):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        log.info('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        log.info('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        # Shift the window by 1 new prediction each time, re-run predictions on new window
        log.info('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        return predicted
