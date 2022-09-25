import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

import os
import re
import csv
import codecs

from keras.models import Model
from keras.layers import Input, Flatten, Concatenate, LSTM, Lambda, Dropout
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import TimeDistributed, Bidirectional, BatchNormalization

from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

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
DATA_PATH = "/content/drive/MyDrive/data/"

GLOVE_DIR = DATA_PATH
TRAIN_DATA_FILE = DATA_PATH + 'quora_train.csv'
TEST_DATA_FILE = DATA_PATH + 'quora_test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.01

def scheduler(epoch, lr):
    if epoch < 4:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def text_to_wordlist(row, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    text = row['question']
    # Convert words to lower case and split them
    if type(text) is str:
        text = text.lower().split()
    else:
        return " "

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

if __name__ == "__main__":

    #load embeddings
    print('Indexing word vectors...')
    embeddings_index = {}
    f = codecs.open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding='utf-8')
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    #load dataset
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df  = pd.read_csv(TEST_DATA_FILE)

    q1df = train_df['question1'].reset_index()
    q2df = train_df['question2'].reset_index()
    q1df.columns = ['index', 'question']
    q2df.columns = ['index', 'question']
    texts_1 = q1df.apply(text_to_wordlist, axis=1, raw=False).tolist()
    texts_2 = q2df.apply(text_to_wordlist, axis=1, raw=False).tolist()
    labels = train_df['is_duplicate'].astype(int).tolist()
    print('Found %s texts.' % len(texts_1))
    del q1df
    del q2df

    q1df = test_df['question1'].reset_index()
    q2df = test_df['question2'].reset_index()
    q1df.columns = ['index', 'question']
    q2df.columns = ['index', 'question']    
    test_texts_1 = q1df.apply(text_to_wordlist, axis=1, raw=False).tolist()
    test_texts_2 = q2df.apply(text_to_wordlist, axis=1, raw=False).tolist()
    test_labels = np.arange(0, test_df.shape[0])
    print('Found %s texts.' % len(test_texts_1))
    del q1df
    del q2df

    #tokenize, convert to sequences and pad
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)
    sequences_1 = tokenizer.texts_to_sequences(texts_1)
    sequences_2 = tokenizer.texts_to_sequences(texts_2)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

    data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
    data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
    labels = np.array(labels)
    print('Shape of data tensor:', data_1.shape)
    print('Shape of label tensor:', labels.shape)

    test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
    test_labels = np.array(test_labels)
    del test_sequences_1
    del test_sequences_2
    del sequences_1
    del sequences_2

    #embedding matrix
    print('Preparing embedding matrix...')
    nb_words = min(MAX_NB_WORDS, len(word_index))

    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    #Multi-Input Architecture
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = Conv1D(128, 3, activation='relu')(embedded_sequences_1)
    x1 = MaxPooling1D(10)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(64, activation='relu')(x1)
    x1 = Dropout(0.2)(x1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = Conv1D(128, 3, activation='relu')(embedded_sequences_2)
    y1 = MaxPooling1D(10)(y1)
    y1 = Flatten()(y1)
    y1 = Dense(64, activation='relu')(y1)
    y1 = Dropout(0.2)(y1)

    merged = Concatenate()([x1, y1])
    merged = BatchNormalization()(merged)
    merged = Dense(64, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)
    preds = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[sequence_1_input,sequence_2_input], outputs=preds)

    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    model.summary()

    #define callbacks
    file_name = SAVE_PATH + 'multi-input-weights-checkpoint.h5'
    checkpoint = ModelCheckpoint(file_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = LearningRateScheduler(scheduler, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=16, verbose=1)
    #tensor_board = TensorBoard(log_dir='./logs', write_graph=True)
    callbacks_list = [checkpoint, reduce_lr, early_stopping]

    hist = model.fit([data_1, data_2], labels, batch_size=1024, epochs=10, callbacks=callbacks_list, validation_split=VALIDATION_SPLIT)

    num_test = 100000
    preds = model.predict([test_data_1[:num_test,:], test_data_2[:num_test,:]])

    quora_submission = pd.DataFrame({"test_id":test_labels[:num_test], "is_duplicate":preds.ravel()})
    quora_submission.to_csv(SAVE_PATH + "quora_submission.csv", index=False)

    plt.figure()
    plt.plot(hist.history['loss'], c='b', lw=2.0, label='train')
    plt.plot(hist.history['val_loss'], c='r', lw=2.0, label='val')
    plt.title('Multi-Input model')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend(loc='upper right')
    plt.show()
    #plt.savefig('./figures/lstm_loss.png')

    plt.figure()
    plt.plot(hist.history['accuracy'], c='b', lw=2.0, label='train')
    plt.plot(hist.history['val_accuracy'], c='r', lw=2.0, label='val')
    plt.title('Multi-Input model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()
    #plt.savefig('./figures/lstm_acc.png')

    plt.figure()
    plt.plot(hist.history['lr'], lw=2.0, label='learning rate')
    plt.title('Multi-Input model')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()
    #plt.savefig('./figures/lstm_learning_rate.png')