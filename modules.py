import torch.nn as nn
import torch
import torch.nn.functional as F
import functional


class inter_joint_reasoning_module(nn.Module):
    def __init__(self, joints,gumbel_temperature,nh=2,use_gumbel_noise=True,hid_dim=256):
        super(inter_joint_reasoning_module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear = nn.Linear(256,joints,bias=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)
        self.one_x_one = torch.nn.Conv2d(joints*hid_dim,256, kernel_size=1)
        self.use_gumbel_noise = use_gumbel_noise
        self.gumbel_temperature = gumbel_temperature

    def forward(self, x,ftr):

        b, j, c, h, w = x.size()

        after_one_one = self.one_x_one(x.view(b,-1,h,w))
        y = torch.relu(self.avg_pool(after_one_one))

        y = self.linear(y.squeeze(-1).squeeze(-1).squeeze(-1))

        y_out = functional.gumbel_sigmoid(y,tau=self.gumbel_temperature,hard=False,use_gumbel_noise=self.use_gumbel_noise).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return ftr * y_out.expand_as(ftr), y_out


