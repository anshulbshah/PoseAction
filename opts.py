import argparse
import sys
import socket
from termcolor import cprint
def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required='True',type=str,help='Dataset to use')
    parser.add_argument('--name',help='Experiment name')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch Size')
    parser.add_argument('--n_epochs', default=150, type=int,help='Number of total epochs to run')
    parser.add_argument('--n_workers',default=8,type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--random_seed',default=0,type=int, help='Random seed to use for training')
    parser.add_argument('--model',default='JMRN',type=str)
    parser.add_argument( '--use_flip', default='True',type=str, help='Use Random Horizontal flip')
    parser.add_argument('--learning_rate',default=0.0001, type=float,help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument( '--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=0.0000001, type=float, help='Weight Decay')
    parser.add_argument('--dropout',default='True',type=str, help='Use dropout')
    parser.add_argument('--normalize',default='True',type=str,help='Normalize')
    parser.add_argument('--paa',default='True', type=str,help='Use PAA')
    parser.add_argument('--paa_type',default='global',type=str,help='PAA type')
    parser.add_argument('--channels',default=3, type=int,help='number of channels')
    parser.add_argument('--max_motion',default=0,type=int,help='Max delta for PAA - global')
    parser.add_argument('--use_background',default="True",type=str,help='Use background channel')
    parser.add_argument('--pose_type',default='openpose',type=str,help='Pose type to use')
    parser.add_argument('--scheduler',default='on_plateau',type=str,help='LR scheduler to use')
    parser.add_argument('--lr_schedule',default='None',type=str,help='LR schedule to use')
    parser.add_argument('--optimizer',default='adam',type=str,help='(adam | SGD)')
    parser.add_argument('--lr_patience',default=5,type=int,help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--save_every', default=5, type=int, help='Save and evaluate frequency')
    parser.add_argument('--loss_weights_contrastive',default=0.0,type=float,help='Loss weight for contrastive')
    parser.add_argument('--gumbel_temperature',default=1,type=float,help='Temperature for gumbel')
    parser.add_argument('--train_split',default='1',type=str,help='Train split to use')
    parser.add_argument('--trainer_type',default='ch_wt_contrastive',type=str)
    parser.add_argument('--compressed_dim',default = 16,type = int)
    parser.add_argument('--normalize_type',default='area',type=str,help='Type of normalization to use')
    parser.add_argument('--tags',default='',type=str,help='experiment tags')
    parser.add_argument('--contrastive_start',default=0,type=int,help='Epoch number to start contrastive loss ')
    parser.add_argument('--contrastive_linear',default=50,type=int,help='Number of epochs to continue for linear increase in loss weight')
    parser.add_argument('--reduced_dim',default=512,type=int)
    parser.add_argument('--max_motion_groupwise',default=0,type=int,help='Max delta for PAA - groupwise')
    parser.add_argument('--gumbel_val',default="False",type=str,help='Use gumbel trick')
    parser.add_argument('--temperature',default=1.0,type=float,help='Temperature for constrastive loss')
    parser.add_argument('--return_augmented_view',default='True',type=str,help='Return augmented view to apply Joint-Contrastive loss')
    parser.add_argument('--contrastive_loss_type',default='all',type=str,help='Type of contrastive loss to use')

    new_sys_argv = []
    for agv in sys.argv:
        if '=' in agv:
            splitted = agv.split('=')
            new_sys_argv.append(splitted[0])
            new_sys_argv.append(splitted[1])
        else:
            new_sys_argv.append(agv)
    sys.argv = new_sys_argv
    args = parser.parse_args()
    if 'alphapose' not in args.pose_type:
        args.num_joints = 19
    return args
