# -*- coding: utf-8 -*-
"""
    @author: AntonOkhotnikov
    Number of speakers classification using the VAD for identifying the speech regions and MFCC
    features to classify the identified frames.
"""


import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
import python_speech_features as psf
import h5py
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Convolution2D as Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import MaxPooling2D, BatchNormalization as BatchNorm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from vad import VAD  # 3rd party library


def get_path():
    """
    :return: path: str: the path to '/16kHz_16bit/' folder
    """
    path = os.getcwd()
    temp = path.split('/')
    temp.pop(-1)
    path = '/'.join(temp)
    path += '/data/16kHz_16bit/'
    return path


def read_files():
    """
    Function yields the dicts with read .wav files containing the signal and sample rate.
    All the files from one folder (same speaker) are merged in one long file.
    """
    for root, dirnames, filenames in os.walk(path):
        arr = np.array([])
        for filename in filenames:
            if filename.endswith('.wav'):
                fs, data = wavfile.read((os.path.join(root, filename)))
                try:
                    arr = np.concatenate((arr, data), axis=0)
                except:
                    arr = data
        try:
            if arr.shape[0] > 0:
                yield {'sample_rate': fs, 'signal': arr}
        except:
            continue


def remove_silence(signal):
    """
    Detects the speech regions and removes the non-speech regions using VAD (Voice Activity Detection)
    :param signal: np.ndarray (n by 1): input audio signal from one speaker
    :return: without_silence: np.ndarray (n by 1): original signal with removed silence regions
    """
    regions = VAD(signal, int(df.sample_rate.iloc[0]), nFFT=512, win_length=0.02, hop_length=0.01, threshold=0.65)

    without_silence = np.array([signal[160 * i: 160 * (i + 1)] for i in range(regions.shape[0]) if regions[i] > 0])
    without_silence = without_silence.flatten()
    return without_silence


def create_by_sum(df, n_voices, n_instances, n_seconds=10):
    """
    Creates a mix of signals and assigns it with number of voices in it by summation
    :param df: pandas DataFrame: DataFrame containing signal and its sample rate
    :param n_voices: int: number of voices we want to mix in one output recording
    :param n_instances: int: how many output recordings we want to produce
    :param n_seconds: float: the length of each output recording
    :return: generator: dictionaries of mixed recording and number of voices in it
    """
    def sum_signals(add_array):
        """
        Adds the signal to the 'mixture' array of mixed signals
        :param add_array: np.ndarray (n by 1): array that adds to the mixture
        :return: pass
        """
        nonlocal mixture
        try:
            if mixture.shape[0] > add_array.shape[0]:
                c = mixture
                c[:add_array.shape[0]] += add_array
                mixture = c
            else:
                c = add_array
                c[:mixture.shape[0]] += mixture
                mixture = c
            pass
        except:
            mixture = add_array
            pass

    for i in range(n_instances):
        sampled = df.sample(n=n_voices)
        mixture = np.array([])
        sampled.speech.apply(lambda arr: sum_signals(arr))

        # yield random part of recording n_seconds long
        length_of_reg = int(n_seconds * sampled.sample_rate.iloc[0])
        rand_reg = np.random.uniform()
        reg_begin = int(rand_reg * (mixture.shape[0] - length_of_reg))
        mixture = mixture[reg_begin: reg_begin + length_of_reg]

        if n_voices <= 4:
            yield {'voices': n_voices, 'signal': mixture}
        else:
            yield {'voices': 4, 'signal': mixture}


def make_features(signal):
    """
    Extracts MFCC features from the window of 0.02 sec with step of 0.01 sec
    :param signal: np.ndarray (n by 1): audio signal
    :return: mfcc: np.ndarray (n/100 by 13): array of MFCC features (1 by 13 vector for each window)
    """
    mfcc = psf.mfcc(signal, samplerate=sample_rate, nfft=1024, nfilt=26, numcep=13, winlen=0.02)
    return mfcc


def merge_n_frames(array_2d, n_merge):
    """
    Merges 'n_merge' frames/windows together
    :param array_2d: np.ndarray (n by 13): MFCC features for 'n' windows of 0.01 sec
    :param n_merge: int: number of frames to merge
    :return: x: np.ndarray (n/n_merge by n_merge by 13): 3-dim array of merged frames
    """
    array_2d = np.reshape(array_2d, (array_2d.shape[0], 1, array_2d.shape[1]))
    x = np.array([])
    for i in range(array_2d.shape[0]//n_merge):
        try:
            x = np.concatenate((x, np.stack(array_2d[i*n_merge: i*n_merge + n_merge], axis=1)), axis=0)
        except:
            x = np.stack(array_2d[i*n_merge: i*n_merge + n_merge], axis=1)
    return x


def extend_target(record):
    """
    Assigns parent's (signal's) label to each merged frame in a row
    :param record: pd.Series: We are interested in column 'voices' (target)
    :return: lst: labels for each merged frame
    """
    return [record.voices for i in range(record.merged.shape[0])]


def save_data(X_train, Y_train):
    """
    Saves the dataset
    :param X_train: np.ndarray (n by m by 13): features
    :param Y_train: np.ndarray (n by 4): target
    :return: pass
    """
    path = os.getcwd()
    temp = path.split('/')
    temp.pop(-1)
    path = '/'.join(temp)
    with h5py.File(path + '/data/train_{frames}frames.h5'.format(frames=n_merge), 'w') as h5file:
        h5file.create_dataset('train', data=X_train)
        h5file.create_dataset('target', data=Y_train)
        h5file.close()


def train_LSTM(X_train, Y_train):
    """
    Builds, trains and evaluates LSTM classifier
    :param X_train: np.ndarray (n by m by 13): features
    :param Y_train: np.ndarray (n by 4): target
    :return: saves the model weights and prints the accuracy and AUC
    """
    # split the set on training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=15)

    np.random.seed(14)  # fix the random numbers generator state

    batch_size = 16
    hidden_units = 10
    input_shape = X_train.shape[1:]
    nb_epochs = 40
    nb_classes = Y_train.shape[1]
    dropout = 0.05
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2, verbose=1)

    model = Sequential()
    model.add(LSTM(units=hidden_units, kernel_initializer='uniform', recurrent_initializer='uniform',
                   dropout=dropout, use_bias=True, unit_forget_bias=True, activation='tanh',
                   recurrent_activation='sigmoid', input_shape=input_shape))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epochs, verbose=1,
              callbacks=[early_stopping], validation_split=0.15)

    print('LSTM classifier performance on the testing set')
    evaluate_model(X_test, Y_test, model)
    model_name = '/data/model_LSTM.h5'
    path = os.getcwd()
    temp = path.split('/')
    temp.pop(-1)
    path = '/'.join(temp)
    model.save(path + model_name)
    print('LSTM classifier is saved', model_name)
    print('-----------------------------------------')


def train_CNN(X_train, Y_train):
    """
    Builds, trains and evaluates CNN classifier
    :param X_train: np.ndarray (n by m by 13): features
    :param Y_train: np.ndarray (n by 4): target
    :return: saves the model weights and prints the accuracy and AUC
    """
    # reshape input for CNN
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
    # split the set on training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=15)

    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)
    batch_size = 16
    nb_epochs = 40
    nb_classes = Y_train.shape[1]
    input_shape = (1, X_train.shape[2], X_train.shape[3])
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2, verbose=1)

    model = Sequential()
    model.add(Conv2D(nb_filters, kernel_size=kernel_size, border_mode='valid', input_shape=input_shape,
                     data_format='channels_first'))
    model.add(BatchNorm())
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2D(nb_filters, kernel_size=kernel_size, data_format='channels_first'))
    model.add(BatchNorm())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=pool_size, strides=(1, 1)))
    model.add(Dropout(0.1))

    model.add(Conv2D(nb_filters, kernel_size=kernel_size, data_format='channels_first'))
    model.add(BatchNorm())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=pool_size, strides=(1, 1)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('linear'))
    model.add(Dropout(0.1))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epochs, verbose=1,
                        callbacks=[early_stopping],
                        validation_split=0.15)

    print('CNN classifier performance on the testing set')
    evaluate_model(X_test, Y_test, model)
    model_name = '/data/model_CNN.h5'
    path = os.getcwd()
    temp = path.split('/')
    temp.pop(-1)
    path = '/'.join(temp)
    model.save(path + model_name)
    print('CNN classifier is saved', model_name)
    print('-----------------------------------------')


def evaluate_model(X_test, Y_test, model):
    """
    Evaluate the performance of a model
    :param X_test: np.ndarray: testing set
    :param Y_test: np.ndarray: training set
    :param model: keras.model: classifier (LSTM or CNN)
    :return: print: accuracy, AUC
    """
    Y_pred = model.predict_proba(X_test)

    ref = np.zeros(shape=(Y_test.shape[0], Y_test.shape[1]))
    i = 0
    for idx in Y_pred.argmax(axis=-1):
        ref[i, idx] = 1
        i += 1

    print('Accuracy is', accuracy_score(Y_test, ref))
    print('AUC is', roc_auc_score(Y_test, Y_pred))


if __name__ == '__main__':

    # read files
    path = get_path()
    df = pd.DataFrame(read_files())
    print('{num} .wav files are read'.format(num=df.shape[0]))

    # remove non-speech regions using VAD
    print('Removing the non-speech regions')
    df['speech'] = df.signal.apply(lambda signal: remove_silence(signal))

    # create training set
    print('Mixing the voices')
    train_samples = pd.concat([pd.DataFrame(create_by_sum(df, 1, 15)), pd.DataFrame(create_by_sum(df, 2, 15)),
                               pd.DataFrame(create_by_sum(df, 3, 15)), pd.DataFrame(create_by_sum(df, 4, 8)),
                               pd.DataFrame(create_by_sum(df, 5, 5))])
    train_samples = train_samples.reset_index(drop=True)

    # extract MFCC features
    print('Extracting the features')
    sample_rate = df.sample_rate.iloc[0]
    train_samples['mfcc'] = train_samples.signal.apply(lambda signal: make_features(signal))

    # merge n_merge frames together
    n_merge = 16
    train_samples['merged'] = train_samples.mfcc.apply(lambda mfcc_arr: merge_n_frames(mfcc_arr, n_merge))

    # create a target variable for each merged frame
    train_samples['target'] = train_samples.apply(lambda row: extend_target(row), axis=1)

    # merge all the rows in one array
    X_train = np.concatenate(train_samples.merged, axis=0)
    Y_train = np.concatenate(train_samples.target, axis=0)

    # make dummy target
    Y_train = pd.get_dummies(Y_train)

    # save data
    print('Saving the data')
    save_data(X_train, Y_train)

    # shuffle both sets
    X_train, Y_train = shuffle(X_train, Y_train)

    # train LSTM classifier
    print('Train LSTM classifier')
    train_LSTM(X_train, Y_train)

    # train CNN classifier
    print('Train CNN classifier')
    train_CNN(X_train, Y_train)