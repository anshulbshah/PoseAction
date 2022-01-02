import glob
import json
import os
import pickle
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from joblib import Parallel, delayed
from scipy import stats
from termcolor import cprint
from tqdm import tqdm

#import plotly.express as px

prev_bs = 0
prior_dist = None

def get_channel_weights(frames,channels=3):
    channel_weights = np.zeros((channels,frames))
    correction = 0
    T_by_C_min_1_b = frames/(channels-1)
    T_by_C_min_1 = np.floor(T_by_C_min_1_b).astype(np.int)
    start_points = np.linspace(0,frames,channels).astype(np.int)
    for window in range(channels - 1):
        len_window = start_points[window+1] - start_points[window]
        correction = 0 if window == channels-2 or channels == 2 else 1
        channel_weights[window,start_points[window] : start_points[window+1]+correction] = np.linspace(1,0,len_window+correction)
        channel_weights[window+1,start_points[window] : start_points[window+1]+correction] = np.linspace(0,1,len_window+correction)
    return channel_weights

def encode_colors_for_potion(pose_heatmaps,channels=2):
    channel_weights = np.zeros((channels,pose_heatmaps.shape[0]))
    correction = 0
    T_by_C_min_1_b = pose_heatmaps.shape[0]/(channels-1)
    T_by_C_min_1 = np.floor(T_by_C_min_1_b).astype(np.int)
    start_points = np.linspace(0,pose_heatmaps.shape[0],channels).astype(np.int)
    for window in range(channels - 1):
        len_window = start_points[window+1] - start_points[window]
        correction = 0 if window == channels-2 or channels == 2 else 1
        channel_weights[window,start_points[window] : start_points[window+1]+correction] = np.linspace(1,0,len_window+correction)
        channel_weights[window+1,start_points[window] : start_points[window+1]+correction] = np.linspace(0,1,len_window+correction)
    row_sums = channel_weights.sum(axis=1)
    #channel_weights = channel_weights / row_sums[:, np.newaxis]
    reshaped_pose = pose_heatmaps.reshape(pose_heatmaps.shape[0],pose_heatmaps.shape[1]*pose_heatmaps.shape[2]*pose_heatmaps.shape[3])
    pose_representation = np.matmul(channel_weights,reshaped_pose).reshape(channels,pose_heatmaps.shape[1],pose_heatmaps.shape[2],pose_heatmaps.shape[3])
    return pose_representation,channel_weights

class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """
    #----------------------------------------------------------------------
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])
        
    #----------------------------------------------------------------------
    def __repr__(self):
        """"""
        attrs = str([x for x in self.__dict__])

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def draw_heatmap(xlabels,ylabels,inputdata):


    fig, ax = plt.subplots()
    im = ax.imshow(inputdata)
    # im = ax.imshow(inputdata,vmin=0,vmax=1.0)
    cbar = ax.figure.colorbar(im, ax=ax,fraction=0.02, pad=0.04)
    cbar.ax.set_ylabel("", rotation=-90, va="bottom")
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(xlabels,fontsize=7)
    ax.set_yticklabels(ylabels,fontsize=7)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center",
             rotation_mode="default")

    fig.tight_layout()
    return fig

def parse_json(keypoint_data):
    number_of_people = len(keypoint_data['people'])
    people_data = []
    overall_conf = []
    for peeps in range(number_of_people):
        joint_locations = np.zeros([18,2])
        joint_confidence = np.zeros([18])
        for j,start_index in zip(range(18),range(0,18*3,3)):
            j_data = keypoint_data['people'][peeps]['pose_keypoints_2d'][start_index:start_index+3]
            joint_locations[j,0] = j_data[0]
            joint_locations[j,1] = j_data[1]
            joint_confidence[j] = j_data[2]
        overall_confidence = np.mean(joint_confidence)
        peep_dict = {
            'joints':joint_locations,
            'joint_conf':joint_confidence,
            'overall_conf':overall_confidence
        }
        people_data.append(peep_dict)
        overall_conf.append(overall_confidence)
    if len(people_data)>0:
        overall_conf, people_data = zip(*sorted(zip(overall_conf, people_data),reverse=True))
    
    return number_of_people,people_data

def rescale_people_maps(pos_data,new_w,img_w,new_h,img_h):
    pos_data_rescaled = pos_data['joints']
    for j in range(18):
        pos_data_rescaled[j,0] = pos_data_rescaled[j,0]*new_w/img_w
        pos_data_rescaled[j,1] = pos_data_rescaled[j,1]*new_h/img_h
    return pos_data_rescaled

def gaussian():
    gaus_x, gaus_y = np.meshgrid(np.linspace(-1,1,3), np.linspace(-1,1,3))
    d = np.sqrt(gaus_x*gaus_x+gaus_y*gaus_y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return g

def final_extract_hmdb(posetype,channels,split,static=False,person_id=False,num_people_map=4,skip_every=0,coloring_type='potion'):
    path = 'metadata/HMDB51'
    heatmap_path = 'data/HMDB-51/openpose_COCO' 
    color_encoder = encode_colors_for_potion

    def parallel_openpose_coco(idx):
        if os.path.exists(os.path.join(save_dir,data['video_name'][idx]+'.npy')):
            print('{} already exists, skipping!'.format(data['video_name'][idx]))
        potion_path_for_video = os.path.join(heatmap_path,data['class_name'][idx],data['video_name'][idx],'heatmaps')
        image_loaded = cv2.imread(os.path.join(potion_path_for_video,'image_' + str(1).zfill(4) + '_pose_heatmaps.png'),cv2.IMREAD_GRAYSCALE)
        if image_loaded.shape[0] != 368:
            print(stop)
            import ipdb; ipdb.set_trace()  # breakpoint 2366f2a9 //
        reshaped = np.transpose(np.reshape(image_loaded,[368,18+1,-1]),[1,0,2])
        hm_h,hm_w = reshaped.shape[1],reshaped.shape[2]
        if hm_h < hm_w:
            new_h = 64
            new_w = int(hm_w*64/hm_h)
        if hm_h >= hm_w:
            new_w = 64
            new_h = int(hm_h*64/hm_w)
        potion_rep = np.zeros([len(data['frames'][idx]),19,new_h,new_w])
        for frame_no_idx,frame_no in enumerate(data['frames'][idx]):
            if skip_every > 0 and frame_no_idx%skip_every != 0:
                #print('skip')
                continue
            image_loaded = cv2.imread(os.path.join(potion_path_for_video,'image_' + str(frame_no + 1).zfill(4) + '_pose_heatmaps.png'),cv2.IMREAD_GRAYSCALE)
            if image_loaded is None:
                print('Image doesnt exist..recheck',idx,frame_no)
            reshaped = np.transpose(np.reshape(image_loaded,[368,18+1,-1]),[1,0,2])
            for idx_hno,h_no in enumerate(ordered_heatmaps):
                potion_rep[frame_no_idx,idx_hno] = cv2.resize(reshaped[h_no],dsize=(new_w,new_h),interpolation=cv2.INTER_CUBIC)
        trajectory,ch_wts = color_encoder(potion_rep,channels)
        np.save(open(os.path.join(save_dir,data['video_name'][idx]+'.npy'),'wb'),trajectory)

    if posetype == 'openpose_COCO':
        types = ['train','test']
        for tp in types:
            ordered_heatmaps = [0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10,18,1]
            data = pickle.load(open(path+'/' + 'hmdb' + '_' + tp + split + '.pkl','rb'))
            save_dir = os.path.join('/'.join(heatmap_path.split('/')[:-1]),'npys_all',posetype + '_' + str(channels) + '_skip_' + str(skip_every) + '_posetype_' + coloring_type)

            Path(save_dir).mkdir(parents=True, exist_ok=True)
            length = len(data['video_name'])
            Parallel(n_jobs=16,verbose=4)(delayed(parallel_openpose_coco)(idx) for idx in tqdm(range(length)))
            print('done',tp)


def final_extract_jhmdb(posetype,channels,split,static=False,skip_every=0):
    path = 'metadata/JHMDB'
    heatmap_path = 'data/JHMDB/openpose_COCO' 
    num_frames_to_sample = 40
    def parallel_openpose_coco(idx):
        if os.path.exists(os.path.join(save_dir,data['video_name'][idx]+'.npy')):
            print('{} already exists, skipping!'.format(data['video_name'][idx]))
            return 0
        potion_path_for_video = os.path.join(heatmap_path,data['class_name'][idx],data['video_name'][idx],'heatmaps')
        image_loaded = cv2.imread(os.path.join(potion_path_for_video,str(1).zfill(5) + '_pose_heatmaps.png'),cv2.IMREAD_GRAYSCALE)
        if image_loaded.shape[0] != 368:
            print(stop)
            import ipdb; ipdb.set_trace()  # breakpoint 2366f2a9 //
        reshaped = np.transpose(np.reshape(image_loaded,[368,18+1,-1]),[1,0,2])
        hm_h,hm_w = reshaped.shape[1],reshaped.shape[2]
        if hm_h < hm_w:
            new_h = 64
            new_w = int(hm_w*64/hm_h)
        if hm_h >= hm_w:
            new_w = 64
            new_h = int(hm_h*64/hm_w)
        potion_rep = np.zeros([len(data['frames'][idx]),19,new_h,new_w])
        timer_start = time.time()
        for frame_no_idx,frame_no in enumerate(data['frames'][idx]):
            if skip_every > 0 and frame_no_idx%skip_every != 0:
                continue
            image_loaded = cv2.imread(os.path.join(potion_path_for_video,str(frame_no+1).zfill(5) + '_pose_heatmaps.png'),cv2.IMREAD_GRAYSCALE)
            if image_loaded is None:
                print('Image doesnt exist..recheck',idx,frame_no)
            reshaped = np.transpose(np.reshape(image_loaded,[368,18+1,-1]),[1,0,2])
            for idx_hno,h_no in enumerate(ordered_heatmaps):
                potion_rep[frame_no_idx,idx_hno] = cv2.resize(reshaped[h_no],dsize=(new_w,new_h),interpolation=cv2.INTER_CUBIC)
        trajectory,ch_wts = encode_colors_for_potion(potion_rep,channels)
        # print(time.time() - timer_start)

        np.save(open(os.path.join(save_dir,data['video_name'][idx]+'.npy'),'wb'),trajectory)



    if posetype == 'openpose_COCO':
        types = ['train','test']
        for tp in types:
            ordered_heatmaps = [0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10,18,1]
            data = pickle.load(open(path+'/' + 'jhmdb' + '_' + tp + split + '.pkl','rb'))
            save_dir = os.path.join('/'.join(heatmap_path.split('/')[:-1]),'npys_all',posetype + '_' + str(channels) + '_skip_' + str(skip_every))
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            length = len(data['video_name'])
            # parallel_openpose_coco(0)
            Parallel(n_jobs=16,verbose=4)(delayed(parallel_openpose_coco)(idx) for idx in tqdm(range(length)))
            print('done',tp)

def final_extract_charades(posetype,channels,split,static=False):
    path = 'metdata/Charades'
    heatmap_path = 'data/Charades/' + posetype 

    def parallel_openpose_coco(idx):
        video_id = data['video_name'][idx]
        potion_path_for_video = os.path.join(heatmap_path,video_id,'heatmaps')
        image_loaded = cv2.imread(os.path.join(potion_path_for_video,video_id + '-' + str(1).zfill(6) + '_pose_heatmaps.png'),0)
        if image_loaded.shape[0] != 368:
            print(stop)
            import ipdb; ipdb.set_trace()  # breakpoint 2366f2a9 //
        reshaped = np.transpose(np.reshape(image_loaded,[368,18+1,-1]),[1,0,2])
        hm_h,hm_w = reshaped.shape[1],reshaped.shape[2]
        if hm_h < hm_w:
            new_h = 64
            new_w = int(hm_w*64/hm_h)
        if hm_h >= hm_w:
            new_w = 64
            new_h = int(hm_h*64/hm_w)
        potion_rep = np.zeros([len(data['frames'][idx]),19,new_h,new_w])
        for frame_no_idx,frame_no in enumerate(data['frames'][idx]):
            image_loaded = cv2.imread(os.path.join(potion_path_for_video,video_id + '-' + str(frame_no+1).zfill(6) + '_pose_heatmaps.png'),0)
            if image_loaded is None:
                print('Image doesnt exist..recheck',idx,frame_no)
            reshaped = np.transpose(np.reshape(image_loaded,[368,18+1,-1]),[1,0,2])
            for idx_hno,h_no in enumerate(ordered_heatmaps):
                potion_rep[frame_no_idx,idx_hno] = cv2.resize(reshaped[h_no],dsize=(new_w,new_h),interpolation=cv2.INTER_CUBIC)
        trajectory,ch_wts = encode_colors_for_potion(potion_rep,channels)
        np.save(open(os.path.join(save_dir,data['video_name'][idx]+'.npy'),'wb'),trajectory)

        

    if posetype == 'openpose_COCO' and not static:
        types = ['train','test']
        for tp in types:
            ordered_heatmaps = [0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10,18,1]
            data = pickle.load(open(path+'/' + 'charades' + '_' + tp + '.pkl','rb'))
            save_dir = os.path.join('/'.join(heatmap_path.split('/')[:-1]),'npys_all',posetype + '_' + str(channels))
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            length = len(data['video_name'])
            Parallel(n_jobs=16,verbose=4)(delayed(parallel_openpose_coco)(idx) for idx in tqdm(range(length)))
            print('done',tp)

    
