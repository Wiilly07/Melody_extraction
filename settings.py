import argparse

M = 40
N = 20
model_width = 80
batch_size = 1024
aug = False
model = 'type_b'
fold = 1
epoch = 30
pile = 100
gpu = 0
class_w = 1

# save_folder = f'../records/{model}_{M}_{N}'
# model_file = f'{save_folder}/fold{fold}.h5'
# txt_file = f'{save_folder}/fold{fold}.txt'

# dataset_ls = ['../datasets/h_american_folk/',
#               '../datasets/h_mozart_piano_training/',
#               '../datasets/p_bach_chorales/',
#               '../datasets/p_haydn_sq/']

# dataset_ls = ['../datasets/p_bach_chorales/']

# dataset_portion = (0.5, 1, 1, 4)
dataset_portion = (1, 0, 0)

def get_args():

    my_parser = argparse.ArgumentParser(description='settings of hyperparamaters')
    my_parser.add_argument("--M", type=int, default=M,
                        help="the shape of the note vector")
    my_parser.add_argument("--N", type=int, default=N)
    my_parser.add_argument("--model_width", type=int, default=model_width)
    my_parser.add_argument("--batch_size", type=int, default=batch_size)
    my_parser.add_argument("--model", type=str, default=model)
    my_parser.add_argument("--aug", type=bool, default=aug)
    my_parser.add_argument("--fold", type=int, default=fold)
    my_parser.add_argument("--epoch", type=int, default=epoch)
    my_parser.add_argument("--pile", type=int, default=pile)
    my_parser.add_argument("--save_folder", type=str)
    my_parser.add_argument('--portion', nargs='+', type=float, default=dataset_portion)
    my_parser.add_argument('--gpu', type=int, default=gpu)
    my_parser.add_argument('--class_w', type=float, default=class_w)
    return my_parser.parse_args()


def add_arg(args):
    if args.save_folder is None:
        args.save_folder = f'../records/{args.model}_{args.M}_{args.N}'
    args.model_file = f'{args.save_folder}/fold{args.fold}/'
    args.txt_file = f'{args.save_folder}/fold{args.fold}/log.txt'

