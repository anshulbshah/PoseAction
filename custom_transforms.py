from __future__ import print_function, division
import os
import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)
from torchvision import  utils
import random
import cv2
import utils
from termcolor import cprint



class PotionFlip(object):
    """Flip the potion representation depending on a random number

    Args:
        
    """

    def __init__(self, flip=True,random_threshold=0.5,pose_type='alphapose_new'):
        self.flip = flip
        self.random_threshold = random_threshold
        self.pose_type = pose_type

    def __call__(self, potion_representation):
        
        if self.flip == 'False':
            return potion_representation
        elif 'alphapose' in self.pose_type or 'openpose' in self.pose_type:
            #if np.random.randn() > self.random_threshold:
            if np.random.random() > 0.5:
                #print('flipping')
                if 'openpose_new' in self.pose_type or 'openpose_coco' in self.pose_type:
                    permuted_tensor = potion_representation[:,[0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,17,18]]
                elif 'openpose' in self.pose_type:
                    permuted_tensor = potion_representation[:,[0,1,5,6,7,2,3,4,11,12,13,8,9,10,15,14,17,16]]

                else:
                    permuted_tensor = potion_representation[:,[0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15],:,:]
                flipped = np.ascontiguousarray(np.flip(permuted_tensor,3))
                # print('flipping')
                return flipped
            else:
                # print('not flipping')
                return potion_representation
        elif 'ntu' in self.pose_type:
            # return ['spine_base 0 ','spine_mid 1 ','neck 2','head 3','LSh 4 ','LElb 5','LWr 6','LHand 7',\
            #         'RSh 8','REl 9','RWr 10','RHand 11','LHip 12','LKnee 13','LAnk 14','LFoot 15','RHip 16','RKnee 17',\
            #         'RAnk 18','RFoot 19','Spine 20','LHandTip 21','LThumb 22','RHandTip 23','RThumb 24']
            if np.random.random() > 0.5:
                permuted_tensor = potion_representation[:,[0,1,2,3,8,9,10,11,4,5,6,7,16,17,18,19,12,13,14,15,20,23,24,21,22],:,:]
                flipped = np.ascontiguousarray(np.flip(permuted_tensor,3))
                # print('flipping')
                return flipped
            else:
                # print('not flipping')
                return potion_representation
        elif 'ava_json' in self.pose_type:
            if np.random.random() > 0.5:
                permuted_tensor = potion_representation[:,[0,1,5,6,7,2,3,4,11,12,13,8,9,10,15,14,17,16]]
                flipped = np.ascontiguousarray(np.flip(permuted_tensor,3))
                # print('flipping')
                return flipped
            else:
                return potion_representation
        else:
            raise NotImplementedError


class RandomCrop(object):
    """Crop the given video sequences (t x h x w) at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        c, t, h, w = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h!=th else 0
        j = random.randint(0, w - tw) if w!=tw else 0
        return i, j, th, tw

    def __call__(self, imgs):
        
        if imgs.shape[2] < self.size[0] or imgs.shape[3] < self.size[1]:
            imgs = imgs.transpose(1,2,3,0)
            h,w = self.size
            resize = lambda x: cv2.resize(x, dsize=(w,h))
            out = [resize(img) for img in imgs]
            imgs = np.stack(out).transpose(3,0,1,2)
        else:
            i, j, h, w = self.get_params(imgs, self.size)
            imgs = imgs[:,:, i:i+h, j:j+w]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if imgs.shape[2] < self.size[0] or imgs.shape[3] < self.size[1]:
            imgs = imgs.transpose(1,2,3,0)
            h,w = self.size
            resize = lambda x: cv2.resize(x, dsize=(w,h))
            out = [resize(img) for img in imgs]
            imgs = np.stack(out).transpose(3,0,1,2)
            return imgs
        else:
            c, t, h, w = imgs.shape
            th, tw = self.size
            i = int(np.round((h - th) / 2.))
            j = int(np.round((w - tw) / 2.))
            #print(imgs.shape,i,j,th,tw)
            return imgs[:,:, i:i+th, j:j+tw]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class Normalize_max(object):
    """Flip the potion representation depending on a random number

    Args:
        
    """

    def __init__(self, normalize='True'):
        self.normalize = True if normalize == 'True' else False

    def __call__(self, representation,**kwargs):
        
        if self.normalize:
            channel_joint_max = np.amax(representation,(-1,-2))[:,:,np.newaxis,np.newaxis]
            channel_joint_max[channel_joint_max == 0] = 1.0
            return representation/channel_joint_max
        else:
            return representation/255.0

class Normalize_area(object):
    """Flip the potion representation depending on a random number

    Args:
        
    """

    def __init__(self, normalize='True',channels=3,batch_mode=False):
        self.normalize = True if normalize == 'True' else False
        self.channels = channels
        self.batch_mode = batch_mode

    def __call__(self, representation,**kwargs):
        
        if self.normalize:
            frames = kwargs['frames']
            from_utils = utils.get_channel_weights(frames,self.channels)
            summed = np.sum(from_utils,1)
            
            #print)
            if not self.batch_mode:
                for ch in range(self.channels):
                    representation[ch]/=(summed[ch]*255.0)
            if self.batch_mode:
                for ch in range(self.channels):
                    representation[:,ch]/=(summed[ch]*255.0)

            return representation
        else:
            return representation/255.0

class JointWiseTranslation(object):

    def __init__(self, move_rep = False, max_motion = 11):
        self.move_rep = True if move_rep == 'True' else False
        self.max_motion = max_motion
        self.collection_of_nums = []

    def check_proportion(self):
        as_array = np.asarray(self.collection_of_nums)
        for i in range(self.max_motion):
            print('{} - {}'.format(i,np.where(as_array == i)[0].shape[0]))

    def __call__(self, representation):
        
        if self.move_rep and self.max_motion > 0:
            delta = np.random.randint(1,self.max_motion,(representation.shape[1],2))
            mode =  np.random.randint(0,4,(representation.shape[1],1))
            new_potion_rep = np.zeros_like(representation)
            p_v = 0
            for joint in range(representation.shape[1]):
                
                if representation.shape[1] == 19:
                    if joint == 17:
                        p_v = 1
                    else:
                        p_v = 0
                if mode[joint] == 0:
                    new_potion_rep[:,joint] = np.pad(representation[:,joint,delta[joint,0]:,delta[joint,1]:],pad_width=((0,0),(0,delta[joint,0]),(0,delta[joint,1])),constant_values=p_v)
                elif mode[joint] == 1:
                    new_potion_rep[:,joint] = np.pad(representation[:,joint,delta[joint,0]:,:-delta[joint,1]],pad_width=((0,0),(0,delta[joint,0]),(delta[joint,1],0)),constant_values=p_v)
                elif mode[joint] == 2:
                    new_potion_rep[:,joint] = np.pad(representation[:,joint,:-delta[joint,0],delta[joint,1]:],pad_width=((0,0),(delta[joint,0],0),(0,delta[joint,1])),constant_values=p_v)
                elif mode[joint] == 3:
                    new_potion_rep[:,joint] = np.pad(representation[:,joint,:-delta[joint,0],:-delta[joint,1]],pad_width=((0,0),(delta[joint,0],0),(delta[joint,1],0)),constant_values=p_v)
                else:
                    print('Something wrong')
                    raise NotImplementedError
            return new_potion_rep
                
        else:
            return representation

class GlobalTranslation(object):

    def __init__(self, move_rep = False, max_motion = 11):
        self.move_rep = True if move_rep == 'True' else False
        self.max_motion = max_motion
        self.collection_of_nums = []

    def check_proportion(self):
        as_array = np.asarray(self.collection_of_nums)
        for i in range(self.max_motion):
            print('{} - {}'.format(i,np.where(as_array == i)[0].shape[0]))

    def __call__(self, representation):
        
        if self.move_rep and self.max_motion > 0:
            delta = np.random.randint(1,self.max_motion,(1,2))
            mode =  np.random.randint(0,4,(1,1))
            new_potion_rep = np.zeros_like(representation)
            top_or_bottom = np.random.random()
            for joint in range(representation.shape[1]):
                
                if representation.shape[1] == 19:
                    if joint == 17:
                        p_v = 1
                    else:
                        p_v = 0
                else:
                    p_v = 0
                if mode == 0:
                    new_potion_rep[:,joint] = np.pad(representation[:,joint,delta[0,0]:,delta[0,1]:],pad_width=((0,0),(0,delta[0,0]),(0,delta[0,1])),constant_values=p_v)
                elif mode == 1:
                    new_potion_rep[:,joint] = np.pad(representation[:,joint,delta[0,0]:,:-delta[0,1]],pad_width=((0,0),(0,delta[0,0]),(delta[0,1],0)),constant_values=p_v)
                elif mode == 2:
                    new_potion_rep[:,joint] = np.pad(representation[:,joint,:-delta[0,0],delta[0,1]:],pad_width=((0,0),(delta[0,0],0),(0,delta[0,1])),constant_values=p_v)
                elif mode == 3:
                    new_potion_rep[:,joint] = np.pad(representation[:,joint,:-delta[0,0],:-delta[0,1]],pad_width=((0,0),(delta[0,0],0),(delta[0,1],0)),constant_values=p_v)
                else:
                    print('Something wrong')
                    raise NotImplementedError
            return new_potion_rep
                
        else:
            return representation

class GroupWiseTranslation(object):

    def __init__(self, move_rep=False, max_motion=11):
        self.move_rep = True if move_rep == 'True' else False
        self.max_motion = max_motion
        self.collection_of_nums = []

    def __call__(self, representation):

        if self.move_rep and self.max_motion > 0:

            same_group_dict = {}
            same_group_dict['1'] = [0, 1, 2, 3, 4]
            same_group_dict['2'] = [5, 7, 9]
            same_group_dict['3'] = [6, 8, 10]
            same_group_dict['4'] = [14, 16]
            same_group_dict['5'] = [13, 15]

            new_rep = np.copy(representation)
            for joint_group_key in same_group_dict.keys():
                join_group = same_group_dict[joint_group_key]
                delta = np.random.randint(1, self.max_motion, (1, 2))
                mode = np.random.randint(0, 4, (1, 1))
                for joint in join_group:

                    if representation.shape[1] == 19:
                        if joint == 17:
                            p_v = 1
                        else:
                            p_v = 0
                    elif representation.shape[1] == 18:
                        p_v = 0
                    else:
                        p_v = 0
                    if mode[0] == 0:
                        new_rep[:, joint] = np.pad(
                            representation[:, joint, delta[0, 0]:, delta[0, 1]:],
                            pad_width=((0, 0), (0, delta[0, 0]), (0, delta[0, 1])), constant_values=p_v)
                    elif mode[0] == 1:
                        new_rep[:, joint] = np.pad(
                            representation[:, joint, delta[0, 0]:, :-delta[0, 1]],
                            pad_width=((0, 0), (0, delta[0, 0]), (delta[0, 1], 0)), constant_values=p_v)
                    elif mode[0] == 2:
                        new_rep[:, joint] = np.pad(
                            representation[:, joint, :-delta[0, 0], delta[0, 1]:],
                            pad_width=((0, 0), (delta[0, 0], 0), (0, delta[0, 1])), constant_values=p_v)
                    elif mode[0] == 3:
                        new_rep[:, joint] = np.pad(
                            representation[:, joint, :-delta[0, 0], :-delta[0, 1]],
                            pad_width=((0, 0), (delta[0, 0], 0), (delta[0, 1], 0)), constant_values=p_v)
                    else:
                        print('Something wrong')
                        raise NotImplementedError

            return new_rep

        else:
            return representation