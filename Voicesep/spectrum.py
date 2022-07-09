from sklearn.cluster import SpectralClustering
from .basic_class import Voices
from .basic_class import Song
from tensorflow import keras
import numpy as np
from itertools import permutations

def spectrum_system(song, n, ideal=False, window=40, moving=10, odd=False, excess=10):
    voices = []
    # tracks = []
    if ideal:
        dis_m = song.label_sp_dis_m.copy()
    else:
        dis_m = song.sp_dis_m.copy() 
    
    if len(song) <= window: #如果曲子過短
        cluster = SpectralClustering(n_clusters=n, assign_labels="discretize", affinity='precomputed')
        cluster.fit(dis_m)
        v = Voices(song=song)
        v.read(label=cluster.labels_)
        voices.append(v)
        # tracks.append(predict_cluster_n(top_ev(laplacian(dis_m))))
    else:
        for i in range((len(song)-window)//moving + 2):
            cluster = SpectralClustering(n_clusters=n, assign_labels="discretize", affinity='precomputed')

            if i*moving+window+excess > len(song):
                if odd:
                    if i%2 == 0:  
                        cluster.fit(dis_m[i*moving:, i*moving:])
                        v = Voices(song=song)
                        v.read(start=i*moving, label=cluster.labels_)
                        voices.append(v)
                        break
                    else:
                        cluster.fit(dis_m[i*moving:i*moving+window, i*moving:i*moving+window])
                        v = Voices(song=song)
                        v.read(start=i*moving, label=cluster.labels_)
                        voices.append(v)

                        cluster.fit(dis_m[(i+1)*moving:, (i+1)*moving:])
                        v = Voices(song=song)
                        v.read(start=(i+1)*moving, label=cluster.labels_)
                        voices.append(v)
                        break

                else:
                    cluster.fit(dis_m[i*moving:, i*moving:])
                    v = Voices(song=song)
                    v.read(start=i*moving, label=cluster.labels_)
                    voices.append(v)
                    break

            else:
                cluster.fit(dis_m[i*moving:i*moving+window, i*moving:i*moving+window])

                v = Voices(song=song)
                v.read(start=i*moving, end=i*moving+window-1, label=cluster.labels_)
                voices.append(v)
            # tracks.append(predict_cluster_n(top_ev(laplacian(dis_m[i*moving:i*moving+window, i*moving:i*moving+window]))))
        
    return voices #, tracks

def predict_k(song, window=40, moving=10):
    tracks = []
    dis_m = song.sp_dis_m.copy()
    
    if len(song) <= window:
        tracks.append(predict_cluster_n(top_ev(laplacian(dis_m))))
    else:
        for i in range((len(song)-window)//moving + 2):
            if i*moving+window+5 > len(song):
                tracks.append(predict_cluster_n(top_ev(laplacian(dis_m[i*moving:, i*moving:]))))
                break
            else:
                tracks.append(predict_cluster_n(top_ev(laplacian(dis_m[i*moving:i*moving+window, i*moving:i*moving+window]))))

    return max(set(tracks), key=tracks.count)


def cal_pitch_mean(ls_of_nn, song): # note numbers
    return song.df_multitrack.iloc[ls_of_nn].note.mean()


def sum_by_pitch(ls):

    track_num = len(ls[0])
    V_total_dict = {f'voice_{k}':[] for k in range(track_num)}

    for V in ls:
        V_undict = [i for i in V.voices.values()]
        V_pitch = [cal_pitch_mean(ls_of_nn, V.song) for ls_of_nn in V_undict]
        V_sorted = [x for _, x in sorted(zip(V_pitch, V_undict), key=lambda pair: pair[0])]

        for i, track in enumerate(V_sorted):
            V_total_dict[f'voice_{i}'] += track

    return Voices(song=ls[0].song, Dict=V_total_dict)

def merge(v1, v2, v3):
    def dict_add(v1, v3, v_label_seq):
        return {f'voice_{i}': v1.voices[f'voice_{i}']+v3.voices[f'voice_{j}'] for i, j in enumerate(v_label_seq)}
    def overlap_count(dict_long, dict_short, v_label_seq):
        Sum = 0
        for i, j in enumerate(v_label_seq):
            Sum += len(set(dict_long[f'voice_{i}']) & set(dict_short[f'voice_{j}']))
        return Sum

    p = list(permutations(range(len(v1))))

    list_of_dict = [dict_add(v1=v1, v3=v3, v_label_seq=s) for s in p]
    list_of_score = [
        max([overlap_count(dict_long=dict_long, dict_short=v2.voices, v_label_seq=s) for s in p])
        for dict_long in list_of_dict]
    return Voices(song=v1.song, Dict=list_of_dict[np.argmax(list_of_score)])


def sum_by_connect(ls: 'moving must be half of the window size'):
    V = ls[0]
    for i in range(len(ls)//2):
        V = merge(V, ls[2*i+1], ls[2*i+2])
    return V

def sorted_v_ls(V) -> list:
    V_undict = [i for i in V.voices.values()]
    V_pitch = [cal_pitch_mean(ls_of_nn, V.song) for ls_of_nn in V_undict]
    V_sorted = [x for _, x in sorted(zip(V_pitch, V_undict), key=lambda pair: pair[0])]
    return V_sorted


def sum_by_set(ls: 'moving must be half of the window size', mode):
    # only for k = 2
    melody_set = set([])

    if mode == 'union':
        for V in ls:
            V_sorted = sorted_v_ls(V)
            melody_set = melody_set | set(V_sorted[1])

        V_total_dict = dict()
        V_total_dict['voice_0'] = sorted(melody_set)
        V_total_dict['voice_1'] = sorted(set(range(len(ls[0].song))) - melody_set)

        return Voices(song=ls[0].song, Dict=V_total_dict)

    elif mode == 'inter':
        for i in range(len(ls)-1):
            v1_melody = sorted_v_ls(ls[i])[1]
            v2_melody = sorted_v_ls(ls[i+1])[1]
            inter = set(v1_melody) & set(v2_melody)
            melody_set = melody_set | inter

        melody_set = melody_set | (set(sorted_v_ls(ls[0])[1]) - set(sorted_v_ls(ls[1])[1]))
        melody_set = melody_set | (set(sorted_v_ls(ls[-1])[1]) - set(sorted_v_ls(ls[-2])[1]))

        V_total_dict = dict()
        V_total_dict['voice_0'] = sorted(melody_set)
        V_total_dict['voice_1'] = sorted(set(range(len(ls[0].song))) - melody_set)

        return Voices(song=ls[0].song, Dict=V_total_dict)

    else:
        return None

def sum_by_moving(ls, N, threshold):
    song_len = len(ls[0].song)

    melody_vote = np.zeros(song_len)
    touched = np.zeros(song_len)
    if song_len >= 2*N:
        touched[:] = N
        touched[0:N] = np.arange(1, N+1)
        touched[-N:] = np.arange(1, N+1)[::-1]
    else:
        top = (song_len+1)//2
        touched[0:top] = np.arange(1, top+1)
        touched[-top:] = np.arange(1, top+1)[::-1]

    for V in ls:
        V_sorted = sorted_v_ls(V)
        for note in V_sorted[1]:
            melody_vote[note] += 1


    if threshold == 0:
        melody_notes = list(np.where(melody_vote>0)[0])
    else:
        melody_notes = list(np.where(melody_vote>=touched*threshold)[0])

    V_total_dict = dict()
    V_total_dict['voice_0'] = sorted(melody_notes)
    V_total_dict['voice_1'] = sorted(set(range(song_len)) - set(melody_notes))

    return Voices(song=ls[0].song, Dict=V_total_dict)

        





def pair_sum(ls, filling=False):
    if len(ls) == 1:
        return ls[0]
    else:
        while(1):
            new_ls = [a+b for a, b in zip(ls[0::2], ls[1::2])]
            if filling:
                for v in new_ls:
                    v.filling_missing_notes()
            if len(ls) == 2:
                return new_ls[0]

            if len(ls) % 2 == 1:
                new_ls.append(ls[-1])

            ls = new_ls
        
def pair_sum_step(ls):
    try:
        new_ls = [a+b for a, b in zip(ls[0::2], ls[1::2])]
    except AssertionError:
        try:
            for i in range(len(ls)//2):
                a = ls[0::2][i]+ls[1::2][i]  
        except AssertionError:
            raise AssertionError('index {} {}'.format(i*2, i*2+1))
    if len(ls) % 2 == 1:
        new_ls.append(ls[-1])
            
    return new_ls

def laplacian(M):
    W = M.copy()
    np.fill_diagonal(W, 0)
    Diag = np.sum(W, axis=1)
    D = np.diag(Diag)
    return D - W

def top_ev(L, top=10):
    ev = np.sort(np.linalg.eigvals(L))[:top]
    return ev

def predict_cluster_n(ev):
    return np.argmax(np.diff(ev))+1


def metrics(file_name, model, reading_range, split=2, mode='melody', _filename_included=True):

    s = Song(file_name, reading_range=reading_range)

    if mode == 'melody':
        s.delete_accompaniment()

    elif mode == 'polyphony':
        s.delete_same_note()

    s.get_dis_m(model=model, default_value=0.5, reach=30)
    voices = spectrum_system(s, n=split)
    
    if _filename_included:
        metrices_ls = [s.filename.split('/')[-1]]
    else:
        metrices_ls = []

    try:
        V = pair_sum(voices, filling=True)
        V.delete_same_note()
        metrices_ls.append(V.accuracy)
        
        if mode == 'melody':
            metrices_ls.append(V.precision())
            metrices_ls.append(V.recall())
            metrices_ls.append(V.f1_score())

        return metrices_ls
    
    except:
        print('failed!')
        metrices_ls.append(0.5)
        if mode == 'melody':
            metrices_ls += [0.5]*3

        return metrices_ls