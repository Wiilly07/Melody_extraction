import numpy as np
import pandas as pd
import itertools

def get_notes(song, reading_range, dtype='float16'):
    size = len(song)
    pad_X = np.pad(song.df_multitrack.values[:, :3], ((reading_range, reading_range), (0, 0)), mode='constant')

    indices = [range(j, j+reading_range*2+1) for j in range(size)]
    indices = list(itertools.chain(*indices))

    vectors = pad_X[indices].reshape(size, -1, 3)

    onsets = vectors[:,reading_range,1].reshape(-1, 1)
    masks = (vectors[:, :, 0] != 0)*1

    #time shift
    vectors[:,:,1] = vectors[:,:,1] - onsets*masks

    return vectors.astype(dtype)


def pairing_notes(song, reading_range, max_dist, dtype='float16', onehot=False):
    notes = get_notes(song, reading_range, dtype)
    onsets = song.df_multitrack.start.values
    durations = song.df_multitrack.duration.values
    labels = song.df_multitrack.label.values

    try:
        size = len(song)
        indices1 = [range(0, size-j) for j in range(1, max_dist+1)]
        indices1 = list(itertools.chain(*indices1))
        indices2 = [range(j, size) for j in range(1, max_dist+1)]
        indices2 = list(itertools.chain(*indices2))
        X = np.concatenate([notes[indices1], notes[indices2]], axis=2)

        #beat normalization
        X[:, :, [1, 2, 4, 5]] = X[:, :, [1, 2, 4, 5]]/X[:, reading_range, 2].reshape(-1, 1, 1)



        K = [(onsets[j:] - onsets[:-j])/durations[:-j] for j in range(1, max_dist+1)]
        K = np.concatenate(K, axis=0).astype(dtype)
        
        y = [labels[:-j] == labels[j:] for j in range(1, max_dist+1)]

        if onehot:
            y = np.eye(2)[np.concatenate(y, axis=0) * 1]
        else:
            y = np.concatenate(y, axis=0) * 1

    except IndexError:
        X = np.array([], dtype=dtype).reshape(0, reading_range*2+1, 6)
        K = np.array([], dtype=dtype).reshape(0)
        if onehot:
            y = np.array([], dtype=dtype).reshape(0, 2)
        else:
            y = np.array([], dtype=dtype).reshape(0)
    
    return X, K, y

def pairing_notes_range(song, reading_range: int, dist: tuple, dtype='float16', onehot=False):
    notes = get_notes(song, reading_range, dtype)
    onsets = song.df_multitrack.start.values
    durations = song.df_multitrack.duration.values
    labels = song.df_multitrack.label.values

    try:
        size = len(song)
        indices1 = [range(0, size-j) for j in range(dist[0], dist[1]+1)]
        indices1 = list(itertools.chain(*indices1))
        indices2 = [range(j, size) for j in range(dist[0], dist[1]+1)]
        indices2 = list(itertools.chain(*indices2))
        X = np.concatenate([notes[indices1], notes[indices2]], axis=2)

        #beat normalization
        X[:, :, [1, 2, 4, 5]] = X[:, :, [1, 2, 4, 5]]/X[:, reading_range, 2].reshape(-1, 1, 1)



        K = [(onsets[j:] - onsets[:-j])/durations[:-j] for j in range(dist[0], dist[1]+1)]
        K = np.concatenate(K, axis=0).astype(dtype)
        
        y = [labels[:-j] == labels[j:] for j in range(dist[0], dist[1]+1)]

        if onehot:
            y = np.eye(2)[np.concatenate(y, axis=0) * 1]
        else:
            y = np.concatenate(y, axis=0) * 1

    except IndexError:
        X = np.array([], dtype=dtype).reshape(0, reading_range*2+1, 6)
        K = np.array([], dtype=dtype).reshape(0)
        if onehot:
            y = np.array([], dtype=dtype).reshape(0, 2)
        else:
            y = np.array([], dtype=dtype).reshape(0)
    
    return X, K, y