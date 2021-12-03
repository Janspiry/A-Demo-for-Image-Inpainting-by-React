import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .networks import SNGatedConv2dWithActivation
from .transformer.transformer_method import TransformerBlock
from einops import rearrange, repeat
class BaseNetwork(nn.Module):
  def __init__(self):
    super(BaseNetwork, self).__init__()
  
  def print_network(self):
    if isinstance(self, list):
      self = self[0]
    num_params = 0
    for param in self.parameters():
      num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f million. '
          'To see the architecture, do print(network).'% (type(self).__name__, num_params / 1000000))

  def init_weights(self, init_type='normal', gain=0.02):
    '''
    initialize network's weights
    init_type: normal | xavier | kaiming | orthogonal
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
    '''
    def init_func(m):
      classname = m.__class__.__name__
      if classname.find('InstanceNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
          nn.init.constant_(m.weight.data, 1.0)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)
      elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
          nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
          nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'xavier_uniform':
          nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        elif init_type == 'kaiming':
          nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
          nn.init.orthogonal_(m.weight.data, gain=gain)
        elif init_type == 'none':  # uses pytorch's default init method
          m.reset_parameters()
        else:
          raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)

    self.apply(init_func)

    # propagate to children
    for m in self.children():
      if hasattr(m, 'init_weights'):
        m.init_weights(init_type, gain)
class InpaintGenerator(BaseNetwork):
    def __init__(self, input_dim=4, cnum=16, init_weights=True):
        super(InpaintGenerator, self).__init__()

        self.input_dim = input_dim
        self.cnum = cnum
        self.conv1_downsample = SNGatedConv2dWithActivation(in_channels=input_dim, out_channels=cnum, kernel_size=3, stride=2, padding=1,  activation=nn.ELU())
        self.conv2_downsample = SNGatedConv2dWithActivation(in_channels=cnum, out_channels=cnum*2, kernel_size=3, stride=2, padding=1, activation=nn.ELU())
        self.conv3_downsample = SNGatedConv2dWithActivation(in_channels=cnum*2, out_channels=cnum*4, kernel_size=3, stride=2, padding=1, activation=nn.ELU())
        self.conv4_downsample = SNGatedConv2dWithActivation(in_channels=cnum*4, out_channels=cnum*4, kernel_size=3, stride=2, padding=1, activation=nn.ELU())


        self.trans5 = nn.Sequential(
            TransformerBlock(size=[cnum*4, 16, 16],patch_size=1, MiniTransFormer=[128, 1, 4, 512], use_global=True),
            SNGatedConv2dWithActivation(cnum*4, cnum*4, kernel_size=3, stride=1, padding=1)
        )
        self.trans4 = nn.Sequential(
            TransformerBlock(size=[cnum*8, 32, 32],patch_size=2, MiniTransFormer=[256, 1, 4, 1024], use_global=True),
            SNGatedConv2dWithActivation(cnum*8, cnum*2, kernel_size=3, stride=1, padding=1)
        )
        self.trans3 = nn.Sequential(
            TransformerBlock(size=[cnum*4, 64, 64],patch_size=4, MiniTransFormer=[256, 1, 4, 1024], use_global=True),
            SNGatedConv2dWithActivation(cnum*4, cnum, kernel_size=3, stride=1, padding=1)
        )
        self.trans2 = nn.Sequential(
            TransformerBlock(size=[cnum*2, 128, 128],patch_size=16, MiniTransFormer=[256, 1, 8, 1024], use_global=True),
            SNGatedConv2dWithActivation(cnum*2, cnum*2, kernel_size=3, stride=1, padding=1),
            TransformerBlock(size=[cnum*2, 128, 128],patch_size=8, MiniTransFormer=[256, 1, 8, 1024], use_local=True),
            SNGatedConv2dWithActivation(cnum*2, input_dim, kernel_size=3, stride=1, padding=1)
        )
        self.trans1 = nn.Sequential(
            TransformerBlock(size=[input_dim*2, 256, 256],patch_size=32, MiniTransFormer=[512, 2, 8, 2048], use_global=True),
            SNGatedConv2dWithActivation(input_dim*2, input_dim*2, kernel_size=3, stride=1, padding=1),
            TransformerBlock(size=[input_dim*2, 256, 256],patch_size=16, MiniTransFormer=[512, 2, 8, 2048], use_local=True),
            SNGatedConv2dWithActivation(input_dim*2, input_dim, kernel_size=3, stride=1, padding=1)
        )
 
        self.torgb5 = nn.Sequential(
            SNGatedConv2dWithActivation(cnum*4, 3, kernel_size=1, stride=1, padding=0, activation=None),
            nn.Tanh())
        self.torgb4 = nn.Sequential(
            SNGatedConv2dWithActivation(cnum*2, 3, kernel_size=1, stride=1, padding=0, activation=None),
            nn.Tanh())
        self.torgb3 = nn.Sequential(
            SNGatedConv2dWithActivation(cnum, 3, kernel_size=1, stride=1, padding=0, activation=None),
            nn.Tanh())
        self.torgb2 = nn.Sequential(
            SNGatedConv2dWithActivation(input_dim, 3, kernel_size=1, stride=1, padding=0, activation=None),
            nn.Tanh())
        self.torgb1 = nn.Sequential(
            SNGatedConv2dWithActivation(input_dim, 3, kernel_size=1, stride=1, padding=0, activation=None),
            nn.Tanh())
        if init_weights:
            self.init_weights()
    def forward(self, xin, mask=None):
        x = xin
        x1 = x
        x2 = self.conv1_downsample(x1)
        x3 = self.conv2_downsample(x2)
        x4 = self.conv3_downsample(x3)
        x5 = self.conv4_downsample(x4)

        up_x5 = self.trans5(x5)
        up_x4 = self.trans4(torch.cat([x4, F.interpolate(up_x5, scale_factor=2)], dim=1))
        up_x3 = self.trans3(torch.cat([x3, F.interpolate(up_x4, scale_factor=2)], dim=1))
        up_x2 = self.trans2(torch.cat([x2, F.interpolate(up_x3, scale_factor=2)], dim=1))
        up_x1 = self.trans1(torch.cat([x1, F.interpolate(up_x2, scale_factor=2)], dim=1))

        img5 = self.torgb5(up_x5)
        img4 = self.torgb4(up_x4)
        img3 = self.torgb3(up_x3)
        img2 = self.torgb2(up_x2)
        img1 = self.torgb1(up_x1)
        
        feats = [img1, img2, img3, img4, img5]
        return feats, img1

