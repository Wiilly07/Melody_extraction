import concurrent.futures
import numpy as np
import os
from functools import partial
from sklearn.utils import shuffle

import Voicesep.augmentation as aug
from Voicesep.basic_class import Song
from Voicesep.feature_extraction import pairing_notes

def song_aug(filename, M, N, model_width, augment):
    sample_song = Song(filename)
    
    if augment:
        if sample_song.track_num > 2:
            augmenter = aug.Augmenter(aug_ls='polyphony')
        else:
            augmenter = aug.Augmenter(aug_ls='melody')
        augmenter(sample_song)

    X, K, y = pairing_notes(song=sample_song, reading_range=M, max_dist=N)

    pad = model_width-M
    if pad:
        X = np.pad(X, pad_width=([0, 0], [pad, pad], [0, 0]), mode='constant')
    return X, K, y

def data_pipe(file_list, M, N, model_width, augment):
    func = partial(song_aug, M=M, N=N, model_width=model_width, augment=augment)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(func, file_list)
    results = list(results)
    X_list, K_list, y_list = zip(*results)

    return np.concatenate(X_list, axis=0), np.concatenate(K_list, axis=0), np.concatenate(y_list, axis=0)

def train_one_epoch(model, file_list, batch_size, M, N, model_width, augment, class_w=1):
    X, K, y = data_pipe(file_list=file_list, M=M, N=N, model_width=model_width, augment=augment)
    hist = model.fit([X, K], y, batch_size=batch_size, epochs=1, verbose=2, class_weight={0:1, 1:class_w})
    return hist.history['loss'][0], hist.history['sparse_categorical_accuracy'][0]

def training(model, file_list, batch_size, pile, M, N, model_width, augment, class_w):
    file_list = shuffle(file_list)

    loss_ls = []
    acc_ls = []
    
    chunks = len(file_list)//pile + 1
    for i in range(chunks):
        loss, acc, = train_one_epoch(model=model, file_list=file_list[i*pile:(i+1)*pile], 
                                     M=M, N=N, model_width=model_width,
                                     batch_size=batch_size, augment=augment, class_w=class_w)
        if acc < 0.65 and i != 0:
            print('training earlystop')
            return False

        loss_ls.append(loss)
        acc_ls.append(acc)
    print(f'training loss: {np.mean(loss_ls):.4f} - accuracy: {np.mean(acc_ls):.4f}')

    return True

def evaluate(model, file_list, batch_size, pile, M, N, model_width):
    lens = 0
    accs = 0
    losses = 0

    if len(file_list) % pile == 0:
        chunks = len(file_list)//pile
    else:
        chunks = len(file_list)//pile + 1

    for i in range(chunks):
        X, K, y = data_pipe(file_list=file_list[i*pile:(i+1)*pile], M=M, N=N, model_width=model_width, augment=False)
        result = model.evaluate([X, K], y, batch_size=batch_size, verbose=0)
        lens += len(y)
        losses += result[0] * len(y)
        accs += result[1] * len(y)

    acc = accs / lens
    loss = losses / lens

    print(f'validation - {lens} - loss: {loss:.4f} - accuracy: {acc:.4f}')
    return acc

def k_fold(ls, k, n):
    assert (n <= k) & (n > 0)
    
    batch = len(ls)//k
    test = ls[(n-1)*batch: n*batch]
    train = list(set(ls) - set(test))
    return train, test

def read_filenames(fold):

    def readtxt(name, Type, fold):
        with open(f'../datasets/fold/{name}/{Type}_{fold}.txt') as f:
            lines = f.readlines()
        return [line[:-1] for line in lines]

    trains = [readtxt(name, 'train', fold) for name in ['bach', 'mozart', 'folk']]
    tests = [readtxt(name, 'test', fold) for name in ['bach', 'mozart', 'folk']]
    return trains, tests

def random_portion(ls, p):
    index = int(len(ls) * (p - int(p)))
    new_ls = shuffle(ls)[:index]
    return ls * int(p) + new_ls

def generate_portioned_ls(dataset: 'a list of lists', portion: tuple):
    new_dataset = []
    for file_ls, p in zip(dataset, portion):
        new_dataset.append(random_portion(file_ls, p))
    return new_dataset


def training_npz(model, file_list, batch_size, pile, M, N, model_width):
    file_list = shuffle(file_list)
    loss_ls = []
    acc_ls = []

    for file in file_list:
        npz = np.load(file)
        X, K, y = npz['X'], npz['K'], npz['y']
        pad = model_width-M
        if pad:
            X[:, :pad, :] = 0
            X[:, -pad:, :] = 0

        X, K, y = shuffle(X, K, y)

        for i in range(len(y)//pile+1):
            hist = model.fit([X[i*pile:(i+1)*pile], K[i*pile:(i+1)*pile]], y[i*pile:(i+1)*pile], 
                             batch_size=batch_size, epochs=1, verbose=2)

            loss_ls.append(hist.history['loss'][0]) 
            acc_ls.append(hist.history['sparse_categorical_accuracy'][0])

            if hist.history['sparse_categorical_accuracy'][0] < 0.65 and i != 0:
                print('training earlystop')
                return False

    print(f'training loss: {np.mean(loss_ls):.4f} - accuracy: {np.mean(acc_ls):.4f}')

    return True

def evaluate_npz(model, file_list, batch_size, pile, M, N, model_width):
    lens = 0
    accs = 0
    losses = 0

    for file in file_list:
        npz = np.load(file)
        X, K, y = npz['X'], npz['K'], npz['y']
        pad = model_width-M
        if pad:
            X[:, :pad, :] = 0
            X[:, -pad:, :] = 0

        for i in range(len(y)//pile+1):
            result = model.evaluate([X[i*pile:(i+1)*pile], K[i*pile:(i+1)*pile]], y[i*pile:(i+1)*pile], 
                                    batch_size=batch_size, verbose=0)

            lens += len(y[i*pile:(i+1)*pile])
            losses += result[0] * len(y[i*pile:(i+1)*pile])
            accs += result[1] * len(y[i*pile:(i+1)*pile])

    acc = accs / lens
    loss = losses / lens

    print(f'validation - {lens} - loss: {loss:.4f} - accuracy: {acc:.4f}')
    return acc
