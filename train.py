import torch
import numpy as np
import torch.optim as optim
import os

from torch.utils.data import DataLoader
from opts import parse_opts
import json
import pdb
import wandb
from dataloaders import *
from models import get_model
import torch.nn as nn
from evaluation import *
import trainers
import itertools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

opts = parse_opts()
extra_args = {}
if opts.tags != "":
    extra_args['tags'] = opts.tags
if opts.name is not None:
    extra_args['name'] = opts.name
wandb.init(project="PoseAction",**extra_args)
wandb.run.save()
opts.name = wandb.run.name
wandb.config.update(opts)

set_random_seeds(opts.random_seed)
collate_fn_train = None
collate_fn_test = None

def collate_ava(batch):
    output_batch = {}
    output_batch['label'] = [torch.Tensor(b['label']) for b in batch]
    output_batch['label'] = torch.cat(output_batch['label'])
    output_batch['num_boxes'] = [torch.Tensor(len(b['label'])) for b in batch]
    output_batch['video_name_actual'] = [b['video_name_actual'] for b in batch]
    output_batch['motion_rep'] = [torch.Tensor(b['motion_rep']) for b in batch]
    output_batch['motion_rep'] = torch.cat(output_batch['motion_rep'])
    if 'motion_rep_augmented' in batch[0].keys():
        output_batch['motion_rep_augmented'] = [torch.Tensor(b['motion_rep_augmented']) for b in batch]
        output_batch['motion_rep_augmented'] = torch.cat(output_batch['motion_rep_augmented'])
    output_batch['slowfast_logit'] = [torch.Tensor(b['slowfast_logit']) for b in batch]
    output_batch['slowfast_logit'] = torch.cat(output_batch['slowfast_logit'])
    output_batch['key_name'] = [b['key_name'] for b in batch]

    if 'metadata' in batch[0]:
        data = [b['metadata'] for b in batch]
        output_batch['metadata'] = torch.Tensor(list(itertools.chain(*data))).view(-1, 2)
        output_batch['ori_boxes'] = [torch.cat([torch.zeros((bb['ori_boxes'].shape[0],1)),torch.Tensor(bb['ori_boxes'])],1) for i,bb in enumerate(batch)]
        output_batch['ori_boxes'] = torch.cat(output_batch['ori_boxes'])
    return output_batch    

if(opts.dataset == 'jhmdb'):
    path = 'metadata/JHMDB/'
    trainDataset = JHMDB(data_loc=path,pose_encoding_path='data/JHMDB/',file_name='jhmdb_train'+opts.train_split,opts=opts,transform_type='train')
    valDataset = JHMDB(data_loc=path,pose_encoding_path='data/JHMDB/',file_name='jhmdb_test'+opts.train_split,opts=opts,transform_type='val')
    valDataset_whole = None
    opts.number_of_classes = 21
elif(opts.dataset == 'hmdb'):
    path = 'metadata/HMDB51/'
    trainDataset = HMDB(data_loc=path,pose_encoding_path='data/HMDB51/',file_name='hmdb_train'+opts.train_split,opts=opts,transform_type='train')
    valDataset = HMDB(data_loc=path,pose_encoding_path='data/HMDB51/',file_name='hmdb_test'+opts.train_split,opts=opts,transform_type='val')
    valDataset_whole = HMDB(data_loc=path,pose_encoding_path='data/HMDB51/',file_name='hmdb_test'+opts.train_split,opts=opts,transform_type='val',get_whole_video=True)
    opts.number_of_classes = 51
elif(opts.dataset == 'charades'):
    path = 'metadata/Charades/'
    trainDataset = Charades(data_loc=path,pose_encoding_path='data/Charades/',file_name='charades_train',opts=opts,transform_type='train')
    valDataset = Charades(data_loc=path,pose_encoding_path='data/Charades/',file_name='charades_test',opts=opts,transform_type='val')
    valDataset_whole = Charades(data_loc=path,pose_encoding_path='data/Charades/',file_name='charades_test',opts=opts,transform_type='val',get_whole_video=True)
    opts.number_of_classes = 157
else:
    raise NotImplementedError
    
print("{:<20} {:<15}".format('Key','Value'))
for k, v in vars(opts).items():
    if(v is None):
        continue
    print("{:<20} {:<15}".format(k, v))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using {}'.format(device))

experiment_name = opts.name

trainLoader = DataLoader(trainDataset,batch_size=opts.batch_size,shuffle=True,num_workers=opts.n_workers,drop_last = False,pin_memory=True, collate_fn=collate_fn_train)
valLoader = DataLoader(valDataset,batch_size=opts.batch_size,shuffle=False,num_workers=opts.n_workers,pin_memory=True, collate_fn=collate_fn_test)
valLoader_whole = None
if valDataset_whole is not None:
    valLoader_whole = DataLoader(valDataset_whole,batch_size=1,shuffle=False,num_workers=0)

if not os.path.exists(opts.dataset):
    os.makedirs(opts.dataset)
if not os.path.exists(os.path.join(opts.dataset,experiment_name)):
    os.makedirs(os.path.join(opts.dataset,experiment_name))

with open(os.path.join(opts.dataset,opts.name,'commandline_args.txt'), 'w') as f:
    json.dump(opts.__dict__, f, indent=2)
    
model = get_model(opts,device)
model = nn.DataParallel(model)
model.to(device)


if (opts.optimizer == 'adam'):
    optimizer = optim.Adam(model.parameters(),opts.learning_rate)
elif opts.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(),lr=opts.learning_rate,momentum = opts.momentum,weight_decay=opts.weight_decay) 

if opts.scheduler == 'on_plateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=opts.lr_patience,verbose=True,factor=0.1)
elif opts.scheduler == 'MultiStepLR':
    lr_schedule = (opts.lr_schedule).split(',')
    lr_schedule = [int(stp) for stp in lr_schedule]
    cprint('Using LR schedule {}'.format(lr_schedule),'yellow')
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_schedule)

temperature_schedule = None

if opts.trainer_type == 'ch_wt_contrastive':
    trainers.default_trainer_ch_wts_contrastive(opts,model,valLoader,trainLoader,device,optimizer,temperature_schedule,scheduler)
