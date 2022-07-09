import sys
import os
from settings import get_args, add_arg


if __name__ == '__main__':

    args = get_args()

    add_arg(args)
    print("the settings")
    print(f'M: {args.M}, N: {args.N}, model_width: {args.model_width}, batch_size: {args.batch_size}')
    print(f'model: {args.model}, fold: {args.fold}, pile: {args.pile}, epoch: {args.epoch}')
    print(f'folder: {args.save_folder}, gpu: {args.gpu}')
    command = f'CUDA_VISIBLE_DEVICES={args.gpu} python training.py'
    # command = f'python training.py'

    for string in sys.argv[1:]:
        command = command + ' ' + string
    
    for loop in range(args.epoch):
        print(f'epoch {loop}')
        os.system(command)