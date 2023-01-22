import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation

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

    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1).astype("float32") / 255
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype("float32") / 255

    y_train_label = keras.utils.to_categorical(y_train)
    y_test_label = keras.utils.to_categorical(y_test)
    num_classes = y_train_label.shape[1]

    #training parameters
    batch_size = 128
    num_epochs = 8

    #model parameters
    num_filters_l1 = 32
    num_filters_l2 = 64

    #CNN architecture
    cnn = Sequential()
    #CONV -> RELU -> MAXPOOL
    cnn.add(Conv2D(num_filters_l1, kernel_size = (5, 5), input_shape=(img_rows, img_cols, 1), padding='same'))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #CONV -> RELU -> MAXPOOL
    cnn.add(Conv2D(num_filters_l2, kernel_size = (5, 5), padding='same'))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #FC -> RELU
    cnn.add(Flatten())
    cnn.add(Dense(128))
    cnn.add(Activation('relu'))

    #Softmax Classifier
    cnn.add(Dense(num_classes))
    cnn.add(Activation('softmax'))

    cnn.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    cnn.summary()

    #define callbacks
    file_name = SAVE_PATH + 'lenet-weights-checkpoint.h5'
    checkpoint = ModelCheckpoint(file_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = LearningRateScheduler(scheduler, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=16, verbose=1)
    #tensor_board = TensorBoard(log_dir='./logs', write_graph=True)
    callbacks_list = [checkpoint, reduce_lr, early_stopping]

    hist = cnn.fit(x_train, y_train_label, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list, validation_split=0.2)

    test_scores = cnn.evaluate(x_test, y_test_label, verbose=2)

    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    y_prob = cnn.predict(x_test)
    y_pred = y_prob.argmax(axis=-1)

    #create submission
    submission = pd.DataFrame(index=pd.RangeIndex(start=1, stop=10001, step=1), columns=['Label'])
    submission['Label'] = y_pred.reshape(-1,1)
    submission.index.name = "ImageId"
    submission.to_csv(SAVE_PATH + '/lenet_pred.csv', index=True, header=True)

    plt.figure()
    plt.plot(hist.history['loss'], 'b', lw=2.0, label='train')
    plt.plot(hist.history['val_loss'], '--r', lw=2.0, label='val')
    plt.title('LeNet model')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend(loc='upper right')
    plt.show()
    #plt.savefig('./figures/lenet_loss.png')

    plt.figure()
    plt.plot(hist.history['accuracy'], 'b', lw=2.0, label='train')
    plt.plot(hist.history['val_accuracy'], '--r', lw=2.0, label='val')
    plt.title('LeNet model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()
    #plt.savefig('./figures/lenet_acc.png')

    plt.figure()
    plt.plot(hist.history['lr'], lw=2.0, label='learning rate')
    plt.title('LeNet model')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()
    #plt.savefig('./figures/lenet_learning_rate.png')
    