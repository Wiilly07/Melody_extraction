import pretty_midi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import itertools
import copy
from itertools import combinations
from itertools import permutations
from collections import namedtuple
from scipy.sparse import diags
from sklearn.metrics import accuracy_score, confusion_matrix, \
                            f1_score, precision_score, recall_score
from .plotting import piano_roll
from .feature_extraction import pairing_notes

def generate_df(midifile) -> 'pd.DataFrame':
    track_list = []
    for index, instrument in enumerate(midifile.instruments): 

        data = [[note.pitch, note.start, note.end] for note in instrument.notes]

        df_singletrack = pd.DataFrame(data, columns=['note', 'start', 'end'])
        df_singletrack['duration'] = df_singletrack['end'] - df_singletrack['start']
        df_singletrack['label'] = index

        track_list.append(df_singletrack)

    df_multitrack = pd.concat(track_list)
    df_multitrack.sort_values(by=['start', 'note', 'duration', 'label'], ascending=[True, True, False, False], inplace=True)
    df_multitrack.reset_index(drop=True, inplace=True)
    df_multitrack.drop(columns='end', inplace=True)
    
    return df_multitrack

# reading_range: note feature的長度 M
# pair_dist: 生出訓練資料時所配對的最長距離 N
# reach: 實際運用時的最長距離

class Song:
    
    def __init__(self, midifile, drum=False):
        
        self.filename = midifile
        mid = pretty_midi.PrettyMIDI(midifile)
        if drum is False:
            mid.instruments = [i for i in mid.instruments if i.is_drum is False]
        
        self.df_multitrack = generate_df(mid)
        self._original_df = self.df_multitrack.copy()

    def reset_df(self):
        self.df_multitrack = self._original_df.copy()   

    @property
    def track_num(self) -> int:
        return len(set(self.df_multitrack.label))
    
    def __len__(self) -> int:
        return len(self.df_multitrack)

    @property
    def voices(self) -> dict:
        _voices = dict()
        for i, label in enumerate(set(self.df_multitrack.label)):
            single_voice = list(self.df_multitrack.loc[self.df_multitrack.label == label].index)

            voice_name = 'voice_{}'.format(i)
            _voices[voice_name] = single_voice

        return _voices

    @property
    def label(self) -> 'pd.Series':
        label_series = pd.Series(data=-1, index=range(len(self)))
        for i, ls in enumerate(self.voices.values()):
            label_series[ls] = i

        return label_series

    def drop_note(self, note):
        self.df_multitrack.drop(note, inplace=True)
        self.df_multitrack.reset_index(inplace=True, drop=True)

    def segment(self, start, end):
        df_segment = self.df_multitrack.loc[start:end].reset_index(drop=True)
        df_segment.start = df_segment.start - df_segment.loc[0].start
        self.df_multitrack = df_segment
             
    def show(self, span=None, mode='note_number'):

        if span is None:
            df = self.df_multitrack

        else:
            if mode == 'note_number':
                df = self.df_multitrack.loc[span[0]:span[1]]
        
        piano_roll(df)
        
    # def _melody_range(self, Range=30, label=1):
    #     index_arr = self.df_multitrack[self.df_multitrack.label==label].index.to_list()
    #     index_arr = [0] + index_arr + [len(self.df_multitrack)]
    #     index_arr = np.array(index_arr)
    #     diff = np.diff(index_arr)
    #     index_tuple = namedtuple('index_pair', ['range', 'index'])
    #     return index_tuple(diff[diff>=Range], index_arr[np.where(diff>=Range)[0]])

    def get_label_dis_m(self, reach, default_value=0.5):

        reach = min(reach, len(self)-1)
        size = len(self)
        mask = diags([1]*(2*reach+1), np.arange(-reach, reach+1), shape=(size, size)).toarray()
        ls = self.df_multitrack.label.to_numpy()
        Dict = {k: (ls == k)*1 for k in set(ls)}
        whole_connected = np.stack([Dict[i] for i in ls])

        dis_m = mask*(whole_connected*0.8+0.1) + (1-mask)*default_value
        np.fill_diagonal(dis_m, 1)

        self.label_dis_m = dis_m

        self.label_sp_dis_m = self.label_dis_m.copy()
        self.label_sp_dis_m[self.label_sp_dis_m > default_value] = 1.0
        self.label_sp_dis_m[self.label_sp_dis_m < default_value] = 1e-6
        self.label_sp_dis_m[self.label_sp_dis_m == default_value] = 1e-3

    def get_dis_m(self, model, reading_range, reach, padding=0, default_value=0.5):

        reach = min(reach, len(self)-1)
        size = len(self)
        dis_m = np.full(shape=(size, size), fill_value=default_value)
        np.fill_diagonal(dis_m, 1)

        pairs, K, __ = pairing_notes(song=self, reading_range=reading_range, max_dist=reach)

        if padding:
            pairs = np.pad(pairs, pad_width=([0, 0], [padding, padding], [0, 0]), mode='constant')

        predicts = model.predict([pairs, K])[:, 1]

        indices1 = [range(0, size-j) for j in range(1, reach+1)]
        indices1 = list(itertools.chain(*indices1))
        indices2 = [range(j, size) for j in range(1, reach+1)]
        indices2 = list(itertools.chain(*indices2))

        dis_m[(indices1, indices2)] = predicts
        dis_m[(indices2, indices1)] = predicts
            
        dis_m[dis_m < 0] = 0.01 
        self.dis_m = dis_m

        self.sp_dis_m = dis_m.copy()
        self.sp_dis_m[self.sp_dis_m > default_value] = 1.0
        self.sp_dis_m[self.sp_dis_m < default_value] = 1e-6
        self.sp_dis_m[self.sp_dis_m == default_value] = 1e-3

        self.get_label_dis_m(default_value=default_value, reach=reach)
        
    def check_same_note(self, tolerance=0.1) -> list:
        df_diff = self.df_multitrack.diff()
        df_diff['duration'] = abs(df_diff['duration']/self.df_multitrack['duration'])
        
        cond1 = (df_diff.note == 0.0)
        cond2 = (df_diff.start == 0.0)
        cond3 = (df_diff.duration <= tolerance) #在tolerance的差距內都算同音
        pairs = [(i-1, i) for i in df_diff[cond1 & cond2 & cond3].index]
        pairs.append(tuple([0]))
        
        fixed_pairs = []
        item = pairs[0]
        for i in pairs[1:]:
            if set(item) & set(i):
                item = tuple(set(item) | set(i))
            else:
                fixed_pairs.append(item)
                item = i
        return fixed_pairs

    def delete_same_note(self):
        twin_notes = self.check_same_note()
        deleted_notes = [twin[0] for twin in twin_notes]
        self.drop_note(deleted_notes)

    def delete_accompaniment(self):
        pair_ls = self.check_same_note()
        label_arr = self.df_multitrack.label.to_numpy()
        candidate_notes = [note for p in pair_ls for note in p]
        def f(pairs, label):
            try:
                return pairs[np.where(label[list(pairs)] == 1)[0][0]] #有主旋律就保留
            except:
                return pairs[0] #非主旋律就隨機保留一個伴奏音
        note_to_keep = [f(pairs=p, label=label_arr) for p in pair_ls]
        note_to_delete = list(set(candidate_notes) - set(note_to_keep))
        self.drop_note(note_to_delete)

    def output_midi(self, filename):
        output_mid = pretty_midi.PrettyMIDI()

        for track in set(self.df_multitrack.label):
            voice_empty = pretty_midi.Instrument(program=0)
            for index, row in self.df_multitrack.loc[self.df_multitrack['label'] == track].iterrows():
                note = pretty_midi.Note(pitch=int(row.note), start=row.start, end=row.start+row.duration, velocity=80)
                voice_empty.notes.append(note)

            output_mid.instruments.append(voice_empty)

        output_mid.write(filename)
       
    def notes(self, reading_range):
        self.reading_range = reading_range
        self.padding_df() #save in self.df_multitrack_pad
        _notes = self.get_notes_feature()
        
        return _notes

    def padding_df(self):
        df_zeropad_1 = pd.DataFrame(data=0, columns=['note','start', 'duration'], 
                                    index=range(-self.reading_range, 0))
        df_zeropad_1['label'] = 'Pad'

        df_zeropad_2 = pd.DataFrame(data=0, columns=['note','start', 'duration'], 
                                    index=range(len(self.df_multitrack), len(self.df_multitrack)+self.reading_range))
        df_zeropad_2['label'] = 'Pad'

        df_pad = pd.concat([df_zeropad_1, self.df_multitrack, df_zeropad_2])
        self.df_multitrack_pad = df_pad
       
    def get_notes_feature(self) -> '#notes by 2M+1 by 3 np.ndarray':
        notes = []

        for i in range(len(self.df_multitrack)):

            notes.append(self.pick_note(i).values)

        notes = np.array(notes)

        return notes
       
    def pick_note(self, index) -> '(2M+1) by 3 pd.Dataframe':
        
        def shift_center(row, center):
            if row['note'] != 0:
                return row['start'] - center

            return 0.0
        
        note = self.df_multitrack_pad.loc[index-self.reading_range:index+self.reading_range, ['note', 'start', 'duration']].copy()
        
        # time shift
        start_center = note.loc[index, 'start']
        note['start'] = note.apply(shift_center, axis=1, center=start_center)

        return note

    def pairing_notes(self, reading_range, max_dist=20, onehot=False):
        notes = self.notes(reading_range)
        onsets = self.df_multitrack.start.values
        durations = self.df_multitrack.duration.values
        labels = self.df_multitrack.label.values

        def beat_norm(note_pair: '(2M+1) by 6 ndarray'):
            duration = note_pair[self.reading_range, 2]
            note_pair[:, [1, 2, 4, 5]] = note_pair[:, [1, 2, 4, 5]]/duration 

        try:
            X = [np.concatenate((notes[:-j], notes[j:]), axis=2) for j in range(1, max_dist+1)]
            K = [(onsets[j:] - onsets[:-j])/durations[:-j] for j in range(1, max_dist+1)]
            y = [labels[:-j] == labels[j:] for j in range(1, max_dist+1)]

            X = np.concatenate(X, axis=0).astype('float32')
            K = np.concatenate(K, axis=0).astype('float32')
            for note_pair in X:
                beat_norm(note_pair)

            if onehot:
                y = np.eye(2)[np.concatenate(y, axis=0) * 1]
            else:
                y = np.concatenate(y, axis=0) * 1

        except IndexError:
            X = np.array([], dtype=np.float32).reshape(0, self.reading_range*2+1, 6)
            K = np.array([], dtype=np.float32).reshape(0)
            if onehot:
                y = np.array([], dtype=np.float32).reshape(0, 2)
            else:
                y = np.array([], dtype=np.float32).reshape(0)
        
        return X, K, y

class Voices:
    def __init__(self, song, Dict=None):
        
        self.song = song
        
        if Dict is None:
            self.voices = dict()
        else:
            self.voices = Dict
            self._voices_copy = copy.deepcopy(Dict)

    def read(self, label, start=0, end=None):
        if end is None:
            end = len(self.song)-1
        
        self.voices = dict()
        sub_df = self.song.df_multitrack.loc[start:end].reset_index()
        
        for i in range(max(label)+1): 
            voice_index = np.where(label == i)[0]
            single_voice = np.ravel(sub_df.loc[voice_index, ['index']].to_numpy()).tolist()

            voice_name = 'voice_{}'.format(i)
            self.voices[voice_name] = single_voice

        outlier_index = np.where(label == -1)[0]
        single_voice = np.ravel(sub_df.loc[outlier_index, ['index']].to_numpy()).tolist()
        self.outlier = single_voice

        self._voices_copy = copy.deepcopy(self.voices)

    def segment(self, start=0, end=None):
        if end is None:
            end = len(self.song)-1
        def list_trim(ls):
            return [i for i in ls if (i >= start) & (i <= end)]

        new_dict = {k:list_trim(i) for (k, i) in self.voices.items()}
        return Voices(song=self.song, Dict=new_dict)
            
    def check(self):
        for v1, v2 in combinations(self.voices.values(), 2):
            assert (set(v1) & set(v2)) == set([]), 'same note in different voices'

    def same_note(self):
        notes = []
        for v1, v2 in combinations(self.voices.values(), 2):
            notes = notes + list(set(v1) & set(v2))
            notes = list(set(notes))

        return notes

    def delete_same_note(self):
        same_notes = self.same_note()
        for note in same_notes:
            keys = [key for key, l in self.voices.items() if note in l]
            keys.remove(random.choice(keys))
            for key in keys:
                self.voices[key].remove(note)
        
    def __len__(self):
        return len(self.voices)

    def length(self):
        Sum = 0
        for k, v in self.voices.items():
            Sum += len(v)
        return Sum
    
    @property
    def _range(self):
        try:
            flat_list = [i for v in self.voices.values() for i in v] + self.outlier
        except:
            flat_list = [i for v in self.voices.values() for i in v]
        return tuple([min(flat_list), max(flat_list)])


    def missing_notes(self, interval=None):
        if interval is None:
            complete_notes = set(range(self._range[0], self._range[1]+1))
        else:
            complete_notes = set(range(interval[0], interval[1]+1))
        self_notes = []
        for v in self.voices.values():
            self_notes += v
        return list(complete_notes - set(self_notes))

    def filling_missing_notes(self, interval=None):
        keys = [key for key in self.voices.keys()]
        for note in self.missing_notes(interval):
            self.voices[random.choice(keys)].append(note)
            
    def reset(self):
        self.voices = copy.deepcopy(self._voices_copy)
    
    def __add__(self, another_V):

        track_num = max(len(self), len(another_V))
        
        if len(another_V) == 0:  
            return self
        
        elif len(self) == 0:
            return another_V
        
        elif len(self) == len(another_V):

            voices = Voices.connect(self, another_V)
                            
            if len(voices) > track_num:
                try:
                    voices.check()
                except AssertionError:
                    voices.reduce_voice2(target_track_num=track_num)
                
            elif len(voices) < track_num:
                if track_num - len(voices) == 1:
                    voices = Voices.connect_2(self, another_V)
                else:
                    raise AssertionError('voice number shrinks')


            # assert (self in voices) & (another_V in voices)
            # voices.delete_same_note()
            return voices
    
    def show(self, track=None):
        
        fig = plt.figure(figsize=(20, 8))
        for index, row in self.song.df_multitrack.iterrows():
            plt.plot([row.start, row.start+row.duration], [row.note, row.note], color='k', alpha=0.6)
            
        colormap = 'brcyk'

        if track is not None:
            track_dict = {k: item for k, item in zip(range(len(self.voices)), self.voices.values())}
            for index in track_dict[track]:
                row = self.song.df_multitrack.loc[index]
                plt.plot([row.start, row.start+row.duration], [row.note, row.note], color='b')
                plt.scatter((2*row.start+row.duration)/2, row.note, s=20, c='b')
        
        else:
            for i, index_list in enumerate(self.voices.values()):
                for index in index_list:
                    row = self.song.df_multitrack.loc[index]
                    plt.plot([row.start, row.start+row.duration], [row.note, row.note], color=colormap[i])
                    plt.scatter((2*row.start+row.duration)/2, row.note, s=20, c=colormap[i])
                            
        plt.show()
        
    @staticmethod
    def connect(voices1, voices2):
        counter = 0
        voices = {}
        for _, v1 in voices1.voices.items():
            for _, v2 in voices2.voices.items():
                if (set(v1) & set(v2)) != set([]):
                    name = 'voice_{}'.format(counter)
                    voices[name] = list(set(v1) | set(v2))
                    voices[name].sort()
                    counter += 1
        return Voices(song=voices1.song, Dict=voices)

    @staticmethod
    def connect_2(voices1, voices2):
        counter = 0
        voices = {}
        v1_tracks = list(range(len(voices1)))
        v2_tracks = list(range(len(voices2)))
        for i, (_, v1) in enumerate(voices1.voices.items()):
            for j, (_, v2) in enumerate(voices2.voices.items()):
                if (set(v1) & set(v2)) != set([]):
                    v1_tracks.remove(i)
                    v2_tracks.remove(j)
                    name = f'voice_{counter}'
                    voices[name] = list(set(v1) | set(v2))
                    voices[name].sort()
                    counter += 1
        name = f'voice_{counter}'
        assert len(v1_tracks) == 1
        assert len(v2_tracks) == 1
        voices[name] = voices1.voices[f'voice_{v1_tracks[0]}'] + voices2.voices[f'voice_{v2_tracks[0]}']
        voices[name].sort()
        return Voices(song=voices1.song, Dict=voices)
    
    def reduce_voice(self, target_track_num):
                
        diff = len(self) - target_track_num
        intersect_num = []
        for v1 in self.voices.values():
            num = 0
            for v2 in self.voices.values():
                if set(v1) != set(v2):
                    num = num + len(set(v1) & set(v2))            
            intersect_num.append(num)
            
        indices = np.array(intersect_num).argsort()[::-1][:diff]
        for index in indices:
            name = 'voice_{}'.format(str(index))
            self.voices.pop(name)

    def reduce_voice2(self, target_track_num):

        def list_concat(Iter):
            total = []
            for i in Iter:
                total = total + i
            return total

        # 要選能夠把所有音符都填滿的track
        cand = [comb for comb in combinations(self.voices.values(), target_track_num) 
                if set(list_concat(comb)) == set(range(self._range[0], self._range[1]+1))] 

        if len(cand) == 1:
            self.voices = {'voice_{}'.format(i): track for i, track in enumerate(cand[0])}
        
        # 沒有track能將音符填滿
        elif len(cand) == 0:
            raise AssertionError('no candidates')

        # 要是有多種選擇，選擇重疊音最少的那個
        else:
            index = np.argmin([sum([len(set(a) & set(b)) for a, b in combinations(k, 2)]) for k in cand])
            self.voices = {'voice_{}'.format(i): track for i, track in enumerate(cand[index])}

    @property
    def wrong_note(self):
        index = self.series.index
        boolmask = self.song.label[index] != self.series
        return self.series[boolmask].index.to_list()

    @property
    def confusion_matrix(self):
        index = self.series.index
        return confusion_matrix(y_true=self.song.label[index], y_pred=self.series)

    def accuracy(self, by):
        weight_dict = {'note': None, 'time': self.song.df_multitrack.duration}
        return round(accuracy_score(y_true=self.song.label, y_pred=self.series, sample_weight=weight_dict[by]), 5)

    # melody line should be label 1
    def precision(self, by):
        return self._metrics(metric='precision', by=by)

    def recall(self, by):
        return self._metrics(metric='recall', by=by)

    def f1_score(self, by):
        return self._metrics(metric='f1_score', by=by)

    def _metrics(self, metric, by):
        assert len(self) == 2
        metrics_dict = {'precision': precision_score, 'recall': recall_score, 'f1_score': f1_score}
        weight_dict = {'note': None, 'time': self.song.df_multitrack.duration}
        func = metrics_dict[metric]
        return round(func(y_true=self.song.label, y_pred=self.series, sample_weight=weight_dict[by]), 5)
    
    @property
    def series(self):
        try:
            return self._series
        except:
            score = 0
            target_dict = dict()
            label_series = pd.Series(data=-1, index=range(len(self.song)))
            for i, ls in enumerate(self.voices.values()):
                label_series[ls] = i
            # label series 是預測voice萃取的答案

            true_label = self.song.label

            # 下面仍可用defaultdict最佳化
            if len(self) >= self.song.track_num:
                for p in permutations(range(len(self)), self.song.track_num):
                    convert_dict = {a: b for a, b in zip(p, range(self.song.track_num))}
                    new_series = label_series.apply(lambda x: convert_dict[x] if x in p else -1)
                    new_score = accuracy_score(true_label, new_series)
                    if new_score >= score:
                        score = new_score
                        target_series = new_series

            elif self.song.track_num > len(self):
                for p in permutations(range(self.song.track_num), len(self)):
                    convert_dict = {a: b for a, b in zip(range(len(self)), p)}
                    new_series = label_series.apply(lambda x: convert_dict[x] if x in range(len(self)) else -1)
                    new_score = accuracy_score(true_label, new_series)
                    if new_score >= score:
                        score = new_score
                        target_series = new_series

            self._series = target_series
            # self._series = target_series[target_series != -1]
            return self._series
        
    def output_midi(self, filename):
        output_mid = pretty_midi.PrettyMIDI()
        for melody in self.voices.values():
            voice_empty = pretty_midi.Instrument(program=0)
            for N in melody:
                row = self.song.df_multitrack.loc[N]
                note = pretty_midi.Note(pitch=int(row.note), start=row.start, end=row.start+row.duration, velocity=80)
                voice_empty.notes.append(note)
                
            output_mid.instruments.append(voice_empty)
            
        output_mid.write(filename)