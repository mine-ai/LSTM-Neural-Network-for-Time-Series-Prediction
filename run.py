from core.utils import plot_results_multiple


import os
import json
import math
from core.data_loader import DataLoader
from core.model import Model
import logging

log_fmt = '%(asctime)s %(levelname)s %(message)s %(name)s.%(funcName)s:%(lineno)d'
logging.basicConfig(
    level=logging.DEBUG,
    format=log_fmt,
    filemode='a',)
log = logging.getLogger()


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
    EPOCHS = 2
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

    x_test_df, y_test_df = data.get_test_data(
        seq_len=SEQUENCE_LENGTH,
        normalise=NORMALISE,
    )

    predictions = model.predict_sequences_multiple(x_test_df, SEQUENCE_LENGTH, SEQUENCE_LENGTH)
    # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    # predictions = model.predict_point_by_point(x_test)

    plot_results_multiple(predictions, y_test_df, SEQUENCE_LENGTH)
    # plot_results(predictions, y_test)


if __name__ == '__main__':
    main()
