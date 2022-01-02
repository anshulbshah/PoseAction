import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import pickle
from modules import *

class PotionClassifier(nn.Module):
    '''
    Simple Encoder1. No Visual information is used here.
    Uses the following network : Trajectory --> LSTM(64,64,1) --> 64 --> 32 --> no_of_classes
    Uses tanh layers
    '''
    def __init__(self,device,opts,extract_features):
        super(PotionClassifier,self).__init__()
        self.number_of_joints = opts.num_joints
        self.number_of_channels = opts.channels
        self.drop_prob = 0.25 if opts.dropout == 'True' else 0.0
        self.indices_to_select = [ni for ni in range(self.number_of_joints)]

        if 'openpose_new' in opts.pose_type and opts.use_background == 'False' and opts.dataset not in ['AVA']:
            self.number_of_joints-=1
            self.indices_to_select.remove(17)
        if opts.use_person_maps == 'True':
            self.number_of_channels = self.number_of_channels+4
        self.conv1 = torch.nn.Conv2d(self.number_of_channels*self.number_of_joints,128, kernel_size=3, stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.drop1 = torch.nn.Dropout2d(self.drop_prob)
        self.conv2 = torch.nn.Conv2d(128,128, kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.drop2 = torch.nn.Dropout2d(self.drop_prob)
        self.conv3 = torch.nn.Conv2d(128,256, kernel_size=3, stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.drop3 = torch.nn.Dropout2d(self.drop_prob)
        self.conv4 = torch.nn.Conv2d(256,256, kernel_size=3, stride=1,padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = torch.nn.Dropout2d(self.drop_prob)
        self.conv5 = torch.nn.Conv2d(256,512, kernel_size=3, stride=2,padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.drop5 = torch.nn.Dropout2d(self.drop_prob)
        self.conv6 = torch.nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.drop6 = torch.nn.Dropout2d(self.drop_prob)
        self.no_class = opts.number_of_classes#49
        self.FC = nn.Linear(512,self.no_class)
        self.device = device
        self.opts = opts
        self.extract_features = extract_features

        #self.plotter = visdom_plot.VisdomLinePlotter(env_name=opts.dataset+'_'+opts.name)
        
    def visualize_fig(self,obs_traj):
        ind_vis = 2
        #zero_dim = torch.zeros((17,1,240,320)).to(self.device)
        # obs_traj = (obs_traj - obs_traj.min())/(obs_traj.max()-obs_traj.min())
        zero_dim = torch.zeros((19,1,64,86)).to(self.device)
        permuted_traj = obs_traj[ind_vis].permute(1,0,2,3)
        #stacked = torch.cat((permuted_traj,zero_dim),dim=1)
        stacked = permuted_traj
        grid_vis = torchvision.utils.make_grid(stacked)
        torchvision.utils.save_image(grid_vis,'checkk.png')

    def visualize_person_map(self,obs_traj):
        ind_vis = 2
        num_person = 4
        #zero_dim = torch.zeros((17,1,240,320)).to(self.device)
        # obs_traj = (obs_traj - obs_traj.min())/(obs_traj.max()-obs_traj.min())
        person_maps = torch.sum(obs_traj[ind_vis,:num_person],1)
        for i in range(num_person):
            torchvision.utils.save_image(person_maps[i],'person_maps' + str(i) + '.png')

        #zero_dim = torch.zeros((19,1,64,86)).to(self.device)
        #permuted_traj = obs_traj[ind_vis].permute(1,0,2,3)
        #stacked = torch.cat((permuted_traj,zero_dim),dim=1)
        #stacked = permuted_traj
        #grid_vis = torchvision.utils.make_grid(stacked)

    def forward(self,obs_traj,sample):
        #print(obs_traj.shape)
        # if(self.opts.use_visdom == 'True'):
        #print(stop)
        #print(obs_traj.shape)
        #self.visualize_fig(obs_traj)
        #obs_traj = obs_traj[:,3:]
        #import pdb; pdb.set_trace()  # breakpoint 60f24b4d //
        
        obs_traj = obs_traj[:,:,self.indices_to_select]
        channel_concat = obs_traj.reshape(obs_traj.shape[0],-1,obs_traj.shape[3],obs_traj.shape[4]) #BS x 57 x 64 x 86
        
        #channel_concat = torch.zeros((32,6*17,64,114)).to(self.device)
        o1 = torch.relu(self.bn1(self.conv1(channel_concat)))  
        o1 = self.drop1(o1) #16 x 128 x 32 x 43

        # print('o1',o1.shape)
        o2 = torch.relu(self.bn2(self.conv2(o1)))
        o2 = self.drop1(o2) #16 x 128 x 32 x 43

        # print('o2',o2.shape)
        o3 = torch.relu(self.bn3(self.conv3(o2)))
        o3 = self.drop1(o3) #16 x 256 x 16 x 22

        # print('o3',o3.shape)
        o4 = torch.relu(self.bn4(self.conv4(o3)))
        o4 = self.drop1(o4) #16 x 256 x 16 x 22

        # print('o4',o4.shape)
        o5 = torch.relu(self.bn5(self.conv5(o4)))
        
        o5 = self.drop1(o5) #16 x 512 x 8 x 11
        
        # print('o5',o5.shape)
        o6 = torch.relu(self.bn6(self.conv6(o5)))
        o6 = self.drop1(o6) #16 x 512 x 8 x 11

        # print('o6',o6.shape)

        # print(o6.shape)
        class_scores = self.FC(torch.mean(torch.mean(o6,dim=-1),dim=-1))
        #print(enc_out.shape,obs_traj.shape)
        #print(stop)
        #final_h = self.maxpool2d(enc_out).squeeze(-1).squeeze(-1).unsqueeze(0)
        #print(final_h.shape)
        #print(stop)
        #print(final_h.shape)
        #print(output.shape)
        #print(stop)
        #final_h = state[0]
        # print(class_scores.shape)
        if self.extract_features:
            return class_scores,None,torch.mean(torch.mean(o6,dim=-1),dim=-1)
        if '2Stream' in self.opts.model:
            return o6
        elif 'sandbox' in self.opts.model:
            return o6
        else:
            return class_scores,None#self.FC4(torch.tanh(self.FC1(final_h))),final_h

class JMRN(nn.Module):
    def __init__(self,device,opts,extract_features):
        super(JMRN,self).__init__()
        self.number_of_joints = opts.num_joints
        self.number_of_channels = opts.channels
        self.drop_prob = 0.25 if opts.dropout == 'True' else 0.0
        self.indices_to_select = [ni for ni in range(self.number_of_joints)]
        if opts.use_background == 'False':
            self.number_of_joints-=1
            self.indices_to_select.remove(17)

        self.input_norm = nn.BatchNorm2d(self.number_of_channels)
        self.conv1 = torch.nn.Conv2d(self.number_of_channels,128, kernel_size=3, stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.drop1 = torch.nn.Dropout2d(self.drop_prob)
        self.conv2 = torch.nn.Conv2d(128,128, kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.drop2 = torch.nn.Dropout2d(self.drop_prob)
        self.conv3 = torch.nn.Conv2d(128,256, kernel_size=3, stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.drop3 = torch.nn.Dropout2d(self.drop_prob)
        self.conv4 = torch.nn.Conv2d(256,256, kernel_size=3, stride=1,padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = torch.nn.Dropout2d(self.drop_prob)
        self.conv_11 = nn.Conv2d(256, opts.compressed_dim, kernel_size=1) 
        self.bn_11 = nn.BatchNorm2d(opts.compressed_dim)
        kwargs = {'use_gumbel_noise':True if opts.gumbel_val == 'True' else False}
        mod_to_use = inter_joint_reasoning_module 
        self.joint_attention_block = mod_to_use(joints=self.number_of_joints,gumbel_temperature=opts.gumbel_temperature,**kwargs)

        self.one_one_conv = torch.nn.Conv2d(self.number_of_joints*opts.compressed_dim,512, kernel_size=1)
        self.conv5 = torch.nn.Conv2d(512,512, kernel_size=3, stride=2,padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.drop5 = torch.nn.Dropout2d(self.drop_prob)
        self.conv6 = torch.nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.drop6 = torch.nn.Dropout2d(self.drop_prob)
        self.no_class = opts.number_of_classes#49

        self.FC = nn.Linear(512,self.no_class)
        self.device = device
        self.opts = opts
        self.batch_norms = []
        for jn in range(self.number_of_joints):
            self.batch_norms.append(nn.BatchNorm2d(self.number_of_channels))
        self.batch_norms = nn.ModuleList(self.batch_norms)
        self.extract_features = extract_features

        self.projection_head = nn.Linear(256,128)
        self.projection_head2 = nn.Linear(128,128)

    def forward(self,obs_traj,sample):
    
        obs_traj = obs_traj[:,:,self.indices_to_select]

        motion_features = []
        for_next_layer = []
        
        for joint in range(obs_traj.shape[2]):
            ch_input = self.batch_norms[joint](obs_traj[:,:,joint])
            o1 = torch.relu(self.conv1(ch_input))
            o1 = self.drop1(o1)
            o2 = torch.relu(self.conv2(o1))
            o2 = self.drop1(o2)
            o3 = torch.relu(self.conv3(o2))
            o3 = self.drop1(o3)
            o4 = torch.relu(self.conv4(o3))
            o4 = self.drop1(o4)
            compressed = torch.relu(self.conv_11(o4))

            motion_features.append(o4)
            for_next_layer.append(compressed)

        input_to_layer = torch.stack(motion_features,1)
        compressed_features = torch.stack(for_next_layer,1)


        new_ftr,channel_weights1 = self.joint_attention_block(input_to_layer,compressed_features)

        new_ftr = new_ftr.view(new_ftr.shape[0],-1,new_ftr.shape[-2],new_ftr.shape[-1])

        channel_weights = channel_weights1.view(channel_weights1.shape[0],channel_weights1.shape[1])

        dim_reduced = torch.relu(self.one_one_conv(new_ftr))
        o5 = torch.relu(self.bn5(self.conv5(dim_reduced)))
        o5 = self.drop1(o5)
        o6 = torch.relu(self.bn6(self.conv6(o5)))
        o6 = self.drop1(o6)

        
        class_scores = self.FC(torch.mean(torch.mean(o6,dim=-1),dim=-1))

        contrast_features = F.normalize(self.projection_head2(torch.relu(self.projection_head(torch.mean(input_to_layer,dim=(3,4))))),p=2,dim=2)

        if self.extract_features:
            return class_scores,channel_weights,torch.mean(torch.mean(o6,dim=-1),dim=-1)
        return class_scores,channel_weights,contrast_features
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model(opts,device,print_num_params = True,extract_features=False):
    #print(opts.model)
    if opts.model == 'PotionClassifier':
        enc = PotionClassifier(device,opts,extract_features)

    elif opts.model == 'JMRN':
        enc = JMRN(device,opts,extract_features)
    else:
        raise NotImplementedError
    if print_num_params:
        print('Number of parameters requiring grad : {0:.2f} Million'.format(count_parameters(enc)/1e6))

    return enc

def save_all_information(name,ep,train_accuracy,val_accuracy,train_loss,val_loss,enc,save_all='False'):
    to_save = {}
    to_save['tr_acc'] = train_accuracy
    to_save['val_acc'] = val_accuracy
    to_save['tr_loss'] = train_loss
    to_save['val_loss'] = val_loss
    with open(name + "/stats.pkl","wb") as fp:
        pickle.dump(to_save,fp)
    #torch.save(enc.state_dict(), name + '/model' + str(ep) + '.t7')
    if(max(val_accuracy) == val_accuracy[-1]):
        print('Saving model')
        torch.save(enc.state_dict(),name + '/model_best.t7')
        return 1
    elif save_all == 'True':
        cprint('Warning saving all models','red')
        torch.save(enc.state_dict(),name + '/model_' + str(ep) + '.t7')
        return 1
    else:
        print('Not saving model')
        return 0
    
    