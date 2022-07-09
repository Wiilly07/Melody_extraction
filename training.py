import os

#os.environ['CUDA_VISIBLE_DIVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import settings
from contextlib import redirect_stdout
from training_utils import training, evaluate, generate_portioned_ls, read_filenames
from models import model_dict

from tensorflow import keras
# import tensorflow as tf

# physical_gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_logical_device_configuration(physical_gpus[0], 
#                        [tf.config.LogicalDeviceConfiguration(memory_limit=1024*20)])


def main():
    args = settings.get_args()
    settings.add_arg(args)

    print(f'M:{args.M}, N:{args.N}, model_width:{args.model_width}, class_w:{args.class_w}')
    print(f'fold:{args.fold}, pile:{args.pile}, batch_size:{args.batch_size}')

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
        for i in range(1, 10):
            os.makedirs(f'{args.save_folder}/fold{i}')

    with open(args.txt_file, 'a') as f:
        with redirect_stdout(f):

            train_ls, val_ls = read_filenames(fold=args.fold)
            train_ls = generate_portioned_ls(train_ls, portion=args.portion) 
            train_ls = [file for ls in train_ls for file in ls] #ravel

            model_files = [file for file in os.listdir(args.model_file) if '.h5' in file]
            if not model_files:
                model_file = f'{args.model_file}/000.h5'
            else:
                model_file = sorted(model_files)[-1]
                model_file = f'{args.model_file}/{model_file}'

            try:
                model = keras.models.load_model(model_file)
                print(f'model_loaded: {model_file}')
                augment = args.aug

            except OSError:
                print('no model, building one')
                model = model_dict[args.model](args.model_width)
                augment = False

            done = training(model=model, file_list=train_ls, 
                            batch_size=args.batch_size, pile=args.pile, 
                            M=args.M, N=args.N, 
                            model_width=args.model_width, augment=augment, class_w=args.class_w)
            
            if done:
                acc_ls = []
                for ls in val_ls:
                    if ls:
                        acc = evaluate(model=model, file_list=ls, 
                                       batch_size=args.batch_size, pile=args.pile, 
                                       M=args.M, N=args.N, 
                                       model_width=args.model_width)
                        acc_ls.append(acc)
                        
                if acc_ls[0] >= 0.85:
                    model.save(f'{args.model_file}/{int(model_file[-6:-3])+1:03d}.h5')
                else:
                    print('validation earlystop')

if __name__ == '__main__':
    main()






