import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np 

import warnings
warnings.filterwarnings("ignore")

from training_utils import training, evaluate, generate_portioned_ls, read_filenames
from models import model_dict

from tensorflow import keras
from Voicesep.basic_class import Song, Voices

from Voicesep.spectrum import spectrum_system, sum_by_pitch, sum_by_moving
from models import model_dict
from tqdm import tqdm
import pandas as pd

def cat_metric_h(file, model, model_width, M, N):
    song = Song(file)
    song.get_dis_m(model=model, reading_range=M, reach=N, padding=model_width-M)
    
    metrics = [file.split('/')[-1]]

    voices = spectrum_system(song, n=2, window=N, moving=1, excess=1)
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        V = sum_by_moving(voices, N=N, threshold=t)
        metrics += [V.precision(by='time'), V.recall(by='time'), V.f1_score(by='time')]

    return metrics

def cat_metirc_p(file, model, model_width, M, N):
    metrics = [file.split('/')[-1]]
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        metrics.append(poly_from_homo_model(file, model, model_width, M, N, threshold=t))
    return metrics

def poly_from_homo_model(file, model, model_width, M, N, threshold):
    label_ls = []
    bach_s = Song(file)
    bach_s.delete_same_note()
    true_id = np.arange(len(bach_s))
    crop_df = bach_s.df_multitrack
    
    
    for _ in range(3):
        bach_s = Song(file)
        bach_s.df_multitrack = crop_df
        bach_s.get_dis_m(model=model, reading_range=M, reach=N, padding=model_width-M)

        voices = spectrum_system(bach_s, n=2, window=N, moving=1, excess=1)
        V_cat = sum_by_moving(voices, N=N, threshold=threshold)

        vn = np.argmin([len(V_cat.voices['voice_0']), len(V_cat.voices['voice_1'])])
        label_ls.append(true_id[V_cat.voices[f'voice_{vn}']].tolist())

        true_id = np.array(list(set(true_id) - set(true_id[V_cat.voices[f'voice_{vn}']])))
        true_id = np.sort(true_id)
        crop_df = bach_s.df_multitrack.loc[V_cat.voices[f'voice_{1-vn}']]
        crop_df = crop_df.reset_index(drop=True, inplace=False)
          
    label_ls.append(true_id.tolist())
    
    voice_dict = {f'voice_{i}': ls for i, ls in enumerate(label_ls)}
    bach_s = Song(file)
    bach_s.delete_same_note()
    V = Voices(song=bach_s, Dict=voice_dict)
    
    return V.accuracy(by='time')

if __name__ == '__main__':
    train_ls, val_ls = read_filenames(fold=1)

    ls = []
    for file in val_ls[0][::3]:
        ls.append('../datasets/p_bach_chorales/'+file.split('/')[-1])


    model_width = 80


    # for M in [20, 40, 60, 80]:
    #     for N in [20, 40, 60]:
    for fold in [1, 2, 3]:
        for M in [50]:
            for N in [20]:
                print(f'M:{M}, N:{N}, fold:{fold}')
                epoch = {20: 40, 40: 20, 60: 14}

                model = model_dict['type_b'](model_width=80)
                try:
                    model.load_weights(f'../records/type_b_{M}_{N}/fold{fold}/{epoch[N]:03d}.h5')
                except:
                    model_dir = f'../records/type_b_{M}_{N}/fold{fold}/'
                    model_files = [file for file in os.listdir(model_dir) if '.h5' in file]
                    model_file = sorted(model_files)[-1]
                    print(f'load {model_file}')
                    model.load_weights(f'../records/type_b_{M}_{N}/fold{fold}/{model_file}')


                bach_csv = f'../evaluation/fold{fold}/{M}_{N}_bach_mov.csv'
                mozart_csv = f'../evaluation/fold{fold}/{M}_{N}_mozart_mov.csv'
                folk_csv = f'../evaluation/fold{fold}/{M}_{N}_folk_mov.csv'

                bach_sheet = []
                for file in tqdm(ls):
                    bach_sheet.append(cat_metirc_p(file, model, model_width, M, N)) 

                df = pd.DataFrame(bach_sheet, columns=['file', 'acc_0', 'acc_0.25', 'acc_0.5', 'acc_0.75', 'acc_1'])
                df.to_csv(bach_csv, index=False)

                mozart_sheet = []
                for file in tqdm(val_ls[1]):
                    mozart_sheet.append(cat_metric_h(file, model, model_width, M, N))

                df = pd.DataFrame(mozart_sheet, columns=['file', 
                                                         'pre_0', 'recall_0', 'f1_0',
                                                         'pre_0.25', 'recall_0.25', 'f1_0.25',
                                                         'pre_0.5', 'recall_0.5', 'f1_0.5',
                                                         'pre_0.75', 'recall_0.75', 'f1_0.75',
                                                         'pre_1', 'recall_1', 'f1_1'])
                df.to_csv(mozart_csv, index=False)

                folk_sheet = []
                for file in tqdm(val_ls[2]):
                    folk_sheet.append(cat_metric_h(file, model, model_width, M, N))

                df = pd.DataFrame(folk_sheet, columns=['file', 
                                                       'pre_0', 'recall_0', 'f1_0',
                                                       'pre_0.25', 'recall_0.25', 'f1_0.25',
                                                       'pre_0.5', 'recall_0.5', 'f1_0.5',
                                                       'pre_0.75', 'recall_0.75', 'f1_0.75',
                                                       'pre_1', 'recall_1', 'f1_1'])
                df.to_csv(folk_csv, index=False)
