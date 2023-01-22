import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler 
from keras.callbacks import EarlyStopping

import math
import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(42)

SAVE_PATH = "/content/drive/MyDrive/Colab Notebooks/data/"

def scheduler(epoch, lr):
    if epoch < 4:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255

    y_train_label = keras.utils.to_categorical(y_train)
    y_test_label = keras.utils.to_categorical(y_test)
    num_classes = y_train_label.shape[1]

    #training params
    batch_size = 64
    num_epochs = 16

    model = Sequential()
    model.add(Dense(128, input_shape=(784, ), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.RMSprop(),
        metrics=["accuracy"]
    )

    model.summary()

    #define callbacks
    file_name = SAVE_PATH + 'mlp-weights-checkpoint.h5'
    checkpoint = ModelCheckpoint(file_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = LearningRateScheduler(scheduler, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=16, verbose=1)
    #tensor_board = TensorBoard(log_dir='./logs', write_graph=True)
    callbacks_list = [checkpoint, reduce_lr, early_stopping]

    hist = model.fit(x_train, y_train_label, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list, validation_split=0.2)

    test_scores = model.evaluate(x_test, y_test_label, verbose=2)

    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    plt.figure()
    plt.plot(hist.history['loss'], 'b', lw=2.0, label='train')
    plt.plot(hist.history['val_loss'], '--r', lw=2.0, label='val')
    plt.title('MLP model')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend(loc='upper right')
    plt.show()
    #plt.savefig('./figures/mlp_loss.png')

    plt.figure()
    plt.plot(hist.history['accuracy'], 'b', lw=2.0, label='train')
    plt.plot(hist.history['val_accuracy'], '--r', lw=2.0, label='val')
    plt.title('MLP model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()
    #plt.savefig('./figures/mlp_acc.png')

    plt.figure()
    plt.plot(hist.history['lr'], lw=2.0, label='learning rate')
    plt.title('MLP model')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()
    #plt.savefig('./figures/mlp_learning_rate.png')