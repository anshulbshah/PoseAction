import os
from torch.utils.data import Dataset
import pickle
import numpy as np
from torchvision import transforms
from custom_transforms import *
from utils import *

class JHMDB(Dataset):
    def __init__(self,data_loc,pose_encoding_path,file_name,opts,transform_type):

        self.file_name = file_name
        self.data = pickle.load(open(data_loc+file_name + '.pkl','rb'))
        self.pose_encoding_path = pose_encoding_path
        if transform_type == 'train':
            use_flip = opts.use_flip
            use_paa = opts.paa
        else:
            use_flip = 'False'
            use_paa = 'False'
        self.flip_potion_transform = PotionFlip(use_flip,0.5,opts.pose_type)
        if opts.normalize_type == 'max':
            self.normalize_transform = Normalize_max(opts.normalize)
        elif opts.normalize_type == 'area':
            self.normalize_transform = Normalize_area(opts.normalize,opts.channels)
        self.normalize_display = transforms.Compose([Normalize_max(opts.normalize),transforms.ToTensor()])

        if opts.paa_type == 'joint_wise':
            self.paa_transform = JointWiseTranslation(use_paa,max_motion=opts.max_motion)
        elif opts.paa_type == 'global':
            self.paa_transform = GlobalTranslation(use_paa,max_motion=opts.max_motion)
        elif opts.paa_type == 'group_wise':
            self.paa_transform = GroupWiseTranslation(use_paa,max_motion=opts.max_motion)
        elif opts.paa_type == 'global_and_groupwise':
            self.paa_transform = transforms.Compose([GlobalTranslation(use_paa,max_motion=opts.max_motion), \
                                                             GroupWiseTranslation(use_paa,max_motion=opts.max_motion_groupwise)])
        if opts.pose_type == 'openpose_coco_v2':
            self.potion_path = f'{self.pose_encoding_path}/openpose_COCO_' + str(opts.channels)
        
        self.channels = opts.channels
        self.opts = opts
        self.transform_type = transform_type
        self.use_flip = opts.use_flip

    def __len__(self):
        return len(self.data['labels'])

    def class_labels(self):
        return self.data['class_labels']

    def joint_names(self):
        return ['Nose','REye','LEye','REar','LEar','RSh','LSh','RElb','LElb','RHand','LHand','RHip','LHip','RKnee','LKnee','RFoot','LFoot','BKG','CNTR']

    def __getitem__(self,idx):
        no_of_frames = len(self.data['frames'][idx])
        potion_path_for_video = os.path.join(self.potion_path,self.data['video_name'][idx])
        trajectory = np.load(potion_path_for_video + '.npy') 
        after_norm = self.normalize_transform(trajectory,frames=no_of_frames)
        after_motion = self.paa_transform(after_norm)
        after_transform = self.flip_potion_transform(after_motion)
        if self.opts.return_augmented_view == 'True':
            after_motion2 = self.paa_transform(after_norm) 
            after_transform2 = self.flip_potion_transform(after_motion2)
            
        label = self.data['labels'][idx]
        sample  = {'label' : label, 
                   'video_name_actual' : self.data['video_name'][idx], 
                   'idx' : idx,
                   'motion_rep':after_transform,
                   'motion_rep_augmented':after_transform2}
        return sample

class HMDB(Dataset):
    def __init__(self,data_loc,pose_encoding_path,file_name,opts,transform_type,get_whole_video=False):

        self.file_name = file_name
        self.data = pickle.load(open(data_loc+file_name + '.pkl','rb'))
        self.pose_encoding_path = pose_encoding_path
        
        if transform_type == 'train':
            use_flip = opts.use_flip
            use_paa = opts.paa
        else:
            use_flip = 'False'
            use_paa = 'False'

        self.random_crop = RandomCrop((64,86))
        self.center_crop = CenterCrop((64,86))

        self.flip_potion_transform = PotionFlip(use_flip,0.5,opts.pose_type)
        if opts.normalize_type == 'max':
            self.normalize_transform = Normalize_max(opts.normalize)
        elif opts.normalize_type == 'area':
            self.normalize_transform = Normalize_area(opts.normalize,opts.channels)
        
        #self.paa_transform = MovePotion(use_paa=opts.paa if transform_type == 'train' else 'False',max_motion=opts.max_motion)
        if opts.paa_type == 'joint_wise':
            self.paa_transform = JointWiseTranslation(use_paa,max_motion=opts.max_motion)
        elif opts.paa_type == 'global':
            self.paa_transform = GlobalTranslation(use_paa,max_motion=opts.max_motion)
        elif opts.paa_type == 'group_wise':
            self.paa_transform = GroupWiseTranslation(use_paa,max_motion=opts.max_motion)
        elif opts.paa_type == 'global_and_groupwise':
            self.paa_transform = transforms.Compose([GlobalTranslation(use_paa,max_motion=opts.max_motion), \
                                                             GroupWiseTranslation(use_paa,max_motion=opts.max_motion_groupwise)])
       
        if opts.pose_type == 'openpose_coco_v2':
            self.potion_path = f'{self.pose_encoding_path}/openpose_COCO_' + str(opts.channels)
        
        self.channels = opts.channels
        self.opts = opts
        self.transform_type = transform_type
        self.use_flip = opts.use_flip
        self.get_whole_video = get_whole_video
        
    def __len__(self):
        return len(self.data['labels'])

    def class_labels(self):
        return self.data['class_labels']

    def joint_names(self):
        return ['Nose','REye','LEye','REar','LEar','RSh','LSh','RElb','LElb','RHand','LHand','RHip','LHip','RKnee','LKnee','RFoot','LFoot','BKG','CNTR']

    def __getitem__(self,idx):
        no_of_frames = len(self.data['frames'][idx])
        potion_path_for_video = os.path.join(self.potion_path,self.data['video_name'][idx])
        trajectory = np.load(potion_path_for_video + '.npy') # Frames x 17 x 64 x 86
        if self.transform_type == 'train':
            trajectory = self.random_crop(trajectory)
        elif self.transform_type == 'val' and not self.get_whole_video:
            trajectory = self.center_crop(trajectory)
        after_norm = self.normalize_transform(trajectory,frames=no_of_frames)
        after_motion = self.paa_transform(after_norm) 
        after_transform = self.flip_potion_transform(after_motion)
        if self.opts.return_augmented_view == 'True':
            after_motion2 = self.paa_transform(after_norm) 
            after_transform2 = self.flip_potion_transform(after_motion2)
       
        label = self.data['labels'][idx]
        sample  = {'label' : label, 
                   'video_name_actual' : self.data['video_name'][idx], 
                   'idx' : idx,
                   'motion_rep':after_transform,
                   'motion_rep_augmented':after_transform2}

        return sample

class Charades(Dataset):
    def __init__(self,data_loc,pose_encoding_path,file_name,opts,transform_type,get_whole_video=False):

        self.file_name = file_name
        self.data = pickle.load(open(data_loc+file_name + '.pkl','rb'))
    
        if transform_type == 'train':
            use_flip = opts.use_flip
            use_paa = opts.paa
        else:
            use_flip = 'False'
            use_paa = 'False'

        self.pose_encoding_path = pose_encoding_path
        self.flip_potion_transform = PotionFlip(use_flip,0.5,opts.pose_type)
        if opts.normalize_type == 'max':
            self.normalize_transform = Normalize_max(opts.normalize)
        elif opts.normalize_type == 'area':
            self.normalize_transform = Normalize_area(opts.normalize,opts.channels)

        self.random_crop = RandomCrop(64)
        self.center_crop = CenterCrop(64)

        if opts.paa_type == 'joint_wise':
            self.paa_transform = JointWiseTranslation(use_paa,max_motion=opts.max_motion)
            cprint('jointwise motion','red')
        elif opts.paa_type == 'global':
            self.paa_transform = GlobalTranslation(use_paa,max_motion=opts.max_motion)
            cprint('global motion','red')
        elif opts.paa_type == 'group_wise':
            self.paa_transform = GroupWiseTranslation(use_paa,max_motion=opts.max_motion)
        elif opts.paa_type == 'global_and_groupwise':
            self.paa_transform = transforms.Compose([GlobalTranslation(use_paa,max_motion=opts.max_motion), \
                                                             GroupWiseTranslation(use_paa,max_motion=opts.max_motion_groupwise)])

        if opts.pose_type == 'openpose_coco':
            self.potion_path = f'{self.pose_encoding_path}/openpose_COCO_' + str(opts.channels)
        
        self.channels = opts.channels
        self.opts = opts
        self.transform_type = transform_type
        self.use_flip = opts.use_flip
        self.get_whole_video = get_whole_video
        self.indices_to_use = list(range(len(self.data['labels'])))



    def __len__(self):
        return len(self.indices_to_use)

    def class_labels(self):
        return self.data['class_labels']

    def joint_names(self):
        return ['Nose','REye','LEye','REar','LEar','RSh','LSh','RElb','LElb','RHand','LHand','RHip','LHip','RKnee','LKnee','RFoot','LFoot','BKG','CNTR']
        
    def __getitem__(self,idx_in):
        idx = self.indices_to_use[idx_in]
        no_of_frames = len(self.data['frames'][idx])
        potion_path_for_video = os.path.join(self.potion_path,self.transform_type,str(idx))
        trajectory = np.load(potion_path_for_video + '.npy') # Frames x 17 x 64 x 86
        if self.transform_type == 'train':
            trajectory_after = self.random_crop(trajectory)
        elif self.transform_type == 'val' and not self.get_whole_video:
            trajectory_after = self.center_crop(trajectory)
        else:
            trajectory_after = trajectory
        after_norm = self.normalize_transform(trajectory_after,frames=no_of_frames)
        after_motion = self.paa_transform(after_norm)
        after_transform = self.flip_potion_transform(after_motion)
        if self.opts.return_augmented_view == 'True':
            after_motion2 = self.paa_transform(after_norm) 
            after_transform2 = self.flip_potion_transform(after_motion2)

        label = self.data['labels'][idx]
        sample  = {     'label' : label[:,0], 
                        'video_name_actual' : self.data['video_name'][idx], 
                        'idx' : idx,
                        'motion_rep':after_transform,
                        'motion_rep_augmented':after_transform2}
        return sample