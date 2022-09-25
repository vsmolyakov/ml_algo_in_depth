import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler 
from keras.callbacks import EarlyStopping

import math
import matplotlib.pyplot as plt

np.random.seed(42)

SAVE_PATH = "/content/drive/MyDrive/Colab Notebooks/data/"

def scheduler(epoch, lr):
    if epoch < 4:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

if __name__ == "__main__":

    img_rows, img_cols = 32, 32
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3).astype("float32") / 255
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3).astype("float32") / 255

    y_train_label = keras.utils.to_categorical(y_train)
    y_test_label = keras.utils.to_categorical(y_test)
    num_classes = y_train_label.shape[1]

    #training parameters
    batch_size = 256
    num_epochs = 32

    #model parameters
    num_filters_l1 = 64
    num_filters_l2 = 128

    #CNN architecture
    cnn = Sequential()
    cnn.add(Conv2D(num_filters_l1, kernel_size = (5, 5), input_shape=(img_rows, img_cols, 3), padding='same'))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    cnn.add(Conv2D(num_filters_l2, kernel_size = (5, 5), padding='same'))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    cnn.add(Flatten())
    cnn.add(Dense(128))
    cnn.add(Activation('relu'))

    cnn.add(Dense(num_classes))
    cnn.add(Activation('softmax'))

    #optimizers
    opt1 = tf.keras.optimizers.SGD()
    opt2 = tf.keras.optimizers.SGD(momentum=0.9, nesterov=True)
    opt3 = tf.keras.optimizers.RMSprop()
    opt4 = tf.keras.optimizers.Adam()

    optimizer_list = [opt1, opt2, opt3, opt4]

    history_list = []

    for idx in range(len(optimizer_list)):

        K.clear_session()

        opt = optimizer_list[idx]

        cnn.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer=opt,
            metrics=["accuracy"]
        )

        #define callbacks
        reduce_lr = LearningRateScheduler(scheduler, verbose=1)
        callbacks_list = [reduce_lr]

        #training loop
        hist = cnn.fit(x_train, y_train_label, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list, validation_split=0.2)
        history_list.append(hist)

    #end for

    plt.figure()
    plt.plot(history_list[0].history['loss'], c='b', lw=2.0, label='SGD')
    plt.plot(history_list[1].history['loss'], c='r', lw=2.0, label='SGD Nesterov')
    plt.plot(history_list[2].history['loss'], c='g', lw=2.0, label='RMSProp')
    plt.plot(history_list[3].history['loss'], c='k', lw=2.0, label='ADAM')
    plt.title('LeNet, CIFAR-100, Optimizers')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Training Loss')
    plt.legend(loc='upper right')
    plt.show()
    #plt.savefig('./figures/lenet_loss.png')

    plt.figure()
    plt.plot(history_list[0].history['val_accuracy'], c='b', lw=2.0, label='SGD')
    plt.plot(history_list[1].history['val_accuracy'], c='r', lw=2.0, label='SGD Nesterov')
    plt.plot(history_list[2].history['val_accuracy'], c='g', lw=2.0, label='RMSProp')
    plt.plot(history_list[3].history['val_accuracy'], c='k', lw=2.0, label='ADAM')
    plt.title('LeNet, CIFAR-100, Optimizers')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend(loc='upper right')
    plt.show()
    #plt.savefig('./figures/lenet_loss.png')

    plt.figure()
    plt.plot(history_list[0].history['lr'], c='b', lw=2.0, label='SGD')
    plt.plot(history_list[1].history['lr'], c='r', lw=2.0, label='SGD Nesterov')
    plt.plot(history_list[2].history['lr'], c='g', lw=2.0, label='RMSProp')
    plt.plot(history_list[3].history['lr'], c='k', lw=2.0, label='ADAM')
    plt.title('LeNet, CIFAR-100, Optimizers')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate Schedule')
    plt.legend(loc='upper right')
    plt.show()
    #plt.savefig('./figures/lenet_loss.png')

    