import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense, Dropout, Activation, Embedding

from keras import regularizers
from keras.preprocessing import sequence
from keras.utils import np_utils

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler 
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

np.random.seed(42)

SAVE_PATH = "/content/drive/MyDrive/Colab Notebooks/data/"

def scheduler(epoch, lr):
    if epoch < 4:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

if __name__ == "__main__":

    #load dataset
    max_words = 20000 # top 20K most frequent words
    seq_len = 200  # first 200 words of each movie review
    (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=max_words)

    x_train = sequence.pad_sequences(x_train, maxlen=seq_len)
    x_val = sequence.pad_sequences(x_val, maxlen=seq_len)

    #training params
    batch_size = 256 
    num_epochs = 8

    #model parameters
    hidden_size = 64
    embed_dim = 128
    lstm_dropout = 0.2
    dense_dropout = 0.5
    weight_decay = 1e-3

    #LSTM architecture
    model = Sequential()
    model.add(Embedding(max_words, embed_dim, input_length=seq_len))
    model.add(Bidirectional(LSTM(hidden_size, dropout=lstm_dropout, recurrent_dropout=lstm_dropout)))
    model.add(Dense(hidden_size, kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
    model.add(Dropout(dense_dropout))
    model.add(Dense(hidden_size/4, kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    model.summary()

    #define callbacks
    file_name = SAVE_PATH + 'lstm-weights-checkpoint.h5'
    checkpoint = ModelCheckpoint(file_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = LearningRateScheduler(scheduler, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=16, verbose=1)
    #tensor_board = TensorBoard(log_dir='./logs', write_graph=True)
    callbacks_list = [checkpoint, reduce_lr, early_stopping]

    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list, validation_data=(x_val, y_val))

    test_scores = model.evaluate(x_val, y_val, verbose=2)

    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    plt.figure()
    plt.plot(hist.history['loss'], c='b', lw=2.0, label='train')
    plt.plot(hist.history['val_loss'], c='r', lw=2.0, label='val')
    plt.title('LSTM model')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend(loc='upper right')
    plt.show()
    #plt.savefig('./figures/lstm_loss.png')

    plt.figure()
    plt.plot(hist.history['accuracy'], c='b', lw=2.0, label='train')
    plt.plot(hist.history['val_accuracy'], c='r', lw=2.0, label='val')
    plt.title('LSTM model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()
    #plt.savefig('./figures/lstm_acc.png')

    plt.figure()
    plt.plot(hist.history['lr'], lw=2.0, label='learning rate')
    plt.title('LSTM model')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()
    #plt.savefig('./figures/lstm_learning_rate.png')

