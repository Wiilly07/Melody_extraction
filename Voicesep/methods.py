from .basic_class import Voices

def skyline(song, mode='poly'):
    df_melody = song.df_multitrack.copy()
    Dict = {}
    index = 0

    while(1):
        onset_list = df_melody.start.unique()
        try:
            pointer = onset_list[0]
        except IndexError:
            # 曲子空了
            break

        voices = []
        while(1):
            note = df_melody[df_melody.start == pointer].note.idxmax()
            voices.append(note)
            note_end = df_melody.loc[note, 'start'] + df_melody.loc[note, 'duration']
            df_melody.drop(labels=note, inplace=True)
            try:
                pointer = onset_list[onset_list >= note_end][0]
            except IndexError:
                # 曲子的盡頭
                break

        Dict['voice_'+str(index)] = voices
        index += 1
        
        if mode == 'melody':
            Dict['voice_1'] = list(set(range(len(song.df_multitrack))) - set(voices))
            break
    
    return Voices(song=song, Dict=Dict)