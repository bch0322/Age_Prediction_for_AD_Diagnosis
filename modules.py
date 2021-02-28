import torch
import torch.nn as nn
import setting_2 as fst
import numpy as np
import utils as ut
import setting as st
import math
from torch.autograd import Function
import torch.nn.functional as F
import nibabel as nib

def MC_dropout(act_vec, p=0.2, mask=True):
    return F.dropout(act_vec, p=p, training=mask, inplace=False)

class LayerNorm(nn.Module):
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.dim))
        self.beta  = nn.Parameter(torch.zeros(cfg.dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class InputDependentCombiningWeights(nn.Module):
    def __init__(self, in_plance, spatial_rank= 1):
        super(InputDependentCombiningWeights, self).__init__()
        """ 1 """
        self.dim_reduction_layer = nn.Conv3d(in_plance, spatial_rank, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        """ 2 """
        self.dilations = [1, 2, 4, 8]
        self.multiscale_layers = nn.ModuleList([])
        for i in range(len(self.dilations)):
            self.multiscale_layers.append(nn.Conv3d(spatial_rank, spatial_rank, kernel_size=3, stride=1, padding=0, dilation=self.dilations[i], groups=spatial_rank, bias=False))

        """ 3 """
        self.squeeze_layer = nn.Sequential(
            nn.Conv3d(spatial_rank * (len(self.dilations) + 2), spatial_rank * 3, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.ReLU(inplace=True),
        )

        """ 4 """
        self.excite_layer = nn.Sequential(
            nn.Conv3d(spatial_rank * 3, spatial_rank * 6, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.Sigmoid(),
        )

        """ 5 """
        self.proj_layer = nn.Conv3d(spatial_rank * 6, spatial_rank, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)

    def forward(self, input_tensor, size):
        x_lowd = self.dim_reduction_layer(input_tensor)  # batch, 16, 85, 105, 80
        x_pool = nn.AvgPool3d(kernel_size=x_lowd.size()[-3:], stride=1)(x_lowd)

        x_multiscale = [
            F.interpolate(x_lowd, size=size, mode='trilinear', align_corners=True),
            F.interpolate(x_pool, size=size, mode='trilinear', align_corners=True),
        ]

        for r, layer in zip(self.dilations, self.multiscale_layers):
            x_multiscale.append(
                F.interpolate(layer(x_lowd), size=size, mode='trilinear', align_corners=True),
            )

        x_multiscale = torch.cat(x_multiscale, 1)
        x_0 = self.squeeze_layer(x_multiscale)
        x_0 = self.excite_layer(x_0)
        x_0 = self.proj_layer(x_0)
        x_0 = nn.Sigmoid()(x_0)
        return x_0

class Input_Dependent_LRLC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups = 1,act_func='relu', bn=True, bias=False, np_feature_map_size = None, n_K = 1):
        super(Input_Dependent_LRLC, self).__init__()
        self.n_K = n_K

        self.cnn_layers = nn.Conv3d(in_channels, out_channels * n_K, kernel_size, stride, padding, dilation, groups=groups, bias=bias)

        self.bn = nn.BatchNorm3d(out_channels) if bn else None
        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func is None:
            self.act_func = None
        else:
            assert False

        self.combining_weights_layer = InputDependentCombiningWeights(in_channels, spatial_rank=n_K)

        #TODO : define the parameters
        self.out_channels = out_channels # out_plane
        self.shape_feature_map = np_feature_map_size # [w, h, d]
        self.dim_feature_map = np_feature_map_size.shape[0] # 3

        """ zero initialized bias vectors for width, height, depth"""
        self.list_parameter_b = nn.ParameterList([])
        for i in range(self.dim_feature_map):
            alpha = nn.Parameter(torch.zeros(np_feature_map_size[i]))
            self.list_parameter_b.append(alpha)
        alpha = nn.Parameter(torch.zeros(out_channels))
        self.list_parameter_b.append(alpha)

    def forward(self, input_tensor):
        """ weight """
        out = self.cnn_layers(input_tensor) # batch, out * rank, w, h, d (6, 64, 42, 52, 39)
        batch_dim = out.shape[0]
        x_dim = out.shape[2]
        y_dim = out.shape[3]
        z_dim = out.shape[4]
        weight = self.combining_weights_layer(input_tensor, size=(x_dim, y_dim, z_dim)) # batch, n_K, 42, 52, 39
        out = out.view(batch_dim, self.out_channels, self.n_K, x_dim, y_dim, z_dim) # batch, f, n_K, w, h, d
        weight = weight.unsqueeze(1)  # batch, 1, n_K, w, h, d
        f_out = torch.sum((out * weight), dim = 2) # batch, f, w, h, d

        """ bias """
        xx_range = self.list_parameter_b[0]
        xx_range = xx_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_dim, 1).view(-1, x_dim)
        xx_range = xx_range[:, None, :, None, None]

        yy_range = self.list_parameter_b[1]
        yy_range = yy_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_dim, 1).view(-1, y_dim)  # (batch, 200)
        yy_range = yy_range[:, None, None, :, None]

        zz_range = self.list_parameter_b[2]
        zz_range = zz_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_dim, 1).view(-1, z_dim)  # (batch, 200)
        zz_range = zz_range[:, None, None, None, :]

        ww_range = self.list_parameter_b[3] # [a]
        ww_range = ww_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_dim, 1).view(-1, self.out_channels)  # (batch, 200)
        ww_range = ww_range[:, :, None, None, None]

        f_out = f_out + xx_range + yy_range + zz_range + ww_range
        f_out = self.bn(f_out)
        f_out = self.act_func(f_out)
        return f_out

class LRLC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups = 1,act_func='leaky', norm_layer='bn', bias=False, np_feature_map_size = None, n_K = 1):
        super(LRLC, self).__init__()
        self.n_K = n_K

        self.cnn_layers = nn.ModuleList([])
        for i in range(n_K):
            self.cnn_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=groups, bias=bias))

        if norm_layer == 'bn':
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm_layer == 'in':
            self.norm = nn.InstanceNorm3d(out_channels)
        elif norm_layer is None:
            self.norm = None
        else:
            assert False

        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func == 'leaky':
            self.act_func = nn.LeakyReLU()
        elif act_func is None:
            self.act_func = None
        else:
            assert False

        #TODO : define the parameters
        self.out_channels = out_channels # out_plane
        self.shape_feature_map = np_feature_map_size # [w, h, d]
        self.dim_feature_map = np_feature_map_size.shape[0] # 3

        """ one initialized weight vectors for width, height, depth"""
        self.list_K = nn.ParameterList([])
        for i in range(self.dim_feature_map):
            alpha = nn.Parameter(torch.ones(self.n_K, np_feature_map_size[i]))
            nn.init.uniform_(alpha, a=0.0, b=1.0)
            self.list_K.append(alpha)

        """ zero initialized bias vectors for width, height, depth"""
        self.list_parameter_b = nn.ParameterList([])
        for i in range(self.dim_feature_map):
            alpha = nn.Parameter(torch.zeros(np_feature_map_size[i]))
            self.list_parameter_b.append(alpha)
        alpha = nn.Parameter(torch.zeros(out_channels))
        self.list_parameter_b.append(alpha)

    def forward(self, input_tensor):
        """ weight """
        f_out = torch.zeros(list(input_tensor.shape[:1]) + [self.out_channels] + list(self.shape_feature_map), dtype=torch.float32).cuda()
        for i in range(self.n_K):
            out = self.cnn_layers[i](input_tensor)
            w_dim = out.shape[1]  # 44
            x_dim = out.shape[2]  # 44
            y_dim = out.shape[3]  # 54
            z_dim = out.shape[4]  # 41
            batch_size_tensor = out.shape[0]
            xx_ones = torch.ones([batch_size_tensor, z_dim], dtype=torch.float32).cuda()
            xx_ones = xx_ones[:, :, None]  # batch, z_dim, 1 (6, 41, 1)
            xx_range = self.list_K[0][i]
            xx_range = xx_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, x_dim)
            xx_range = xx_range[:, None, :] # batch, 1, x_dim
            xx_channel = torch.matmul(xx_ones, xx_range) # batch, z_dim, x_dim
            xx_channel = xx_channel.unsqueeze(3).repeat(1, 1, 1, y_dim).unsqueeze(1).float() # batch, 1, z_dim, x_dim, y_dim
            xx_channel = xx_channel.permute(0, 1, 3, 4, 2) # batch, 1, x, y, z

            yy_ones = torch.ones([batch_size_tensor, x_dim], dtype=torch.float32).cuda()
            yy_ones = yy_ones[:, :, None]  # (batch, 175, 1)
            yy_range = self.list_K[1][i]
            yy_range = yy_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, y_dim)  # (batch, 200)
            yy_range = yy_range[:, None, :]
            yy_channel = torch.matmul(yy_ones, yy_range)
            yy_channel = yy_channel.unsqueeze(3).repeat(1, 1, 1, z_dim).unsqueeze(1).float() # (4, 1, 175, 200, 143)
            yy_channel = yy_channel.permute(0, 1, 2, 3, 4)

            zz_ones = torch.ones([batch_size_tensor, y_dim], dtype=torch.float32).cuda()
            zz_ones = zz_ones[:, :, None]  # (batch, 175, 1)
            zz_range = self.list_K[2][i]
            zz_range = zz_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, z_dim)  # (batch, 200)
            zz_range = zz_range[:, None, :]
            zz_channel = torch.matmul(zz_ones, zz_range)
            zz_channel = zz_channel.unsqueeze(3).repeat(1, 1, 1, x_dim).unsqueeze(1).float() # (4, 1, 175, 200, 143)
            zz_channel = zz_channel.permute(0, 1, 4, 2, 3)

            ## TODO : normalize w matrix
            large_w = (xx_channel + yy_channel + zz_channel)
            # large_w = nn.Softmax(-1)(large_w.contiguous().view(large_w.size()[0], large_w.size()[1], -1)).view_as(large_w)
            large_w = nn.Sigmoid()(large_w)
            f_out += large_w * out

        """ bias """
        xx_range = self.list_parameter_b[0]
        xx_range = xx_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, x_dim)
        xx_range = xx_range[:, None, :, None, None]

        yy_range = self.list_parameter_b[1]
        yy_range = yy_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, y_dim)  # (batch, 200)
        yy_range = yy_range[:, None, None, :, None]

        zz_range = self.list_parameter_b[2]
        zz_range = zz_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, z_dim)  # (batch, 200)
        zz_range = zz_range[:, None, None, None, :]

        ww_range = self.list_parameter_b[3] # [a]
        ww_range = ww_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, w_dim)  # (batch, 200)
        ww_range = ww_range[:, :, None, None, None]

        f_out = f_out + xx_range + yy_range + zz_range + ww_range

        if self.norm is not None:
            f_out = self.norm(f_out)
        if self.act_func is not None:
            f_out = self.act_func(f_out)
        return f_out

class LRLC_2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups = 1,act_func='leaky', norm_layer='bn', bias=False, np_feature_map_size = None, n_K = 1):
        super(LRLC_2, self).__init__()
        self.n_K = n_K

        self.cnn_layers = nn.ModuleList([])
        for i in range(n_K):
            self.cnn_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=groups, bias=bias))

        if norm_layer == 'bn':
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm_layer == 'in':
            self.norm = nn.InstanceNorm3d(out_channels)
        elif norm_layer is None:
            self.norm = None
        else:
            assert False

        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func == 'leaky':
            self.act_func = nn.LeakyReLU()
        elif act_func is None:
            self.act_func = None
        else:
            assert False

        #TODO : define the parameters
        self.out_channels = out_channels # out_plane
        self.shape_feature_map = np_feature_map_size # [w, h, d]
        self.dim_feature_map = np_feature_map_size.shape[0] # 3

        """ one initialized weight vectors for width, height, depth"""
        self.list_K = nn.ParameterList([])
        for i in range(self.n_K):
            alpha = nn.Parameter(torch.ones(tuple(self.shape_feature_map)))
            nn.init.uniform_(alpha, a=0.0, b=1.0)
            self.list_K.append(alpha)


        """ zero initialized bias vectors for width, height, depth"""
        # self.list_parameter_b = nn.ParameterList([])
        # for i in range(self.dim_feature_map):
        #     alpha = nn.Parameter(torch.zeros(np_feature_map_size[i]))
        #     # nn.init.uniform_(alpha, a=0.0, b=1.0)
        #     self.list_parameter_b.append(alpha)
        # alpha = nn.Parameter(torch.zeros(out_channels))
        # self.list_parameter_b.append(alpha)

    def forward(self, input_tensor):
        """ weight """
        f_out = torch.zeros(list(input_tensor.shape[:1]) + [self.out_channels] + list(self.shape_feature_map), dtype=torch.float32).cuda()
        for i in range(self.n_K):
            out = self.cnn_layers[i](input_tensor)
            large_w = self.list_K[i]
            ## TODO : normalize w matrix
            # large_w = nn.Softmax(-1)(large_w.contiguous().view(large_w.size()[0], large_w.size()[1], -1)).view_as(large_w)
            # large_w = nn.Sigmoid()(large_w)
            f_out += large_w * out

        if self.norm is not None:
            f_out = self.norm(f_out)
        if self.act_func is not None:
            f_out = self.act_func(f_out)
        return f_out

class LRLC_RoI(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups = 1,act_func='leaky', norm_layer='bn', bias=False, np_feature_map_size = None, n_K = 1, n_RoI = None, RoI_template = None):
        super(LRLC_RoI, self).__init__()
        self.n_K = n_K
        self.n_RoI = n_RoI
        self.RoI_template = RoI_template
        self.cnn_layers = nn.ModuleList([])
        for i in range(n_K):
            self.cnn_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=groups, bias=bias))

        if norm_layer == 'bn':
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm_layer == 'in':
            self.norm = nn.InstanceNorm3d(out_channels)
        elif norm_layer is None:
            self.norm = None
        else:
            assert False

        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func == 'leaky':
            self.act_func = nn.LeakyReLU()
        elif act_func is None:
            self.act_func = None
        else:
            assert False

        #TODO : define the parameters
        self.out_channels = out_channels # out_plane
        self.shape_feature_map = np_feature_map_size # [w, h, d]
        self.dim_feature_map = np_feature_map_size.shape[0] # 3

        """ one initialized weight vectors for width, height, depth"""
        self.list_K = nn.ParameterList([])
        for i in range(self.n_K):
            alpha = nn.Parameter(torch.ones(self.n_RoI))
            nn.init.uniform_(alpha, a=0.0, b=1.0)
            self.list_K.append(alpha)

        """ zero initialized bias vectors for width, height, depth"""
        self.param_b = nn.Parameter(torch.zeros(self.n_RoI))

    def forward(self, input_tensor):
        """ weight """
        f_out = torch.zeros(list(input_tensor.shape[:1]) + [self.out_channels] + list(self.shape_feature_map), dtype=torch.float32).cuda()
        for i in range(self.n_K):
            out = self.cnn_layers[i](input_tensor)
            large_w = self.list_K[i]

            ## TODO : normalize w matrix
            large_w = nn.Softmax(-1)(large_w)
            for j in range(self.n_RoI):
                out[:, :, (self.RoI_template.squeeze() == j + 1)] *= large_w[j]
            f_out += out

        for j in range(self.n_RoI):
            f_out[:, :, (self.RoI_template.squeeze() == j + 1)] += self.param_b[j]

        if self.norm is not None:
            f_out = self.norm(f_out)
        if self.act_func is not None:
            f_out = self.act_func(f_out)
        return f_out

class BasicConv_Block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, act_func='leaky', norm_layer='in', bias=False):
        super(BasicConv_Block, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='replicate',dilation=dilation, groups=groups, bias=bias)

        if norm_layer == 'bn':
            self.bn = nn.BatchNorm3d(out_planes)
        elif norm_layer == 'in':
            self.bn = nn.InstanceNorm3d(out_planes, affine=False)
        elif norm_layer is None:
            self.bn = None
        else:
            assert False

        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func == 'leaky':
            self.act_func = nn.LeakyReLU()
        elif act_func == 'gelu':
            self.act_func = nn.GELU()
        elif act_func is None:
            self.act_func = None
        else:
            assert False

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act_func is not None:
            x = self.act_func(x)
        return x
class DoubleConv_Block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DoubleConv_Block, self).__init__()
        self.FE = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='replicate', dilation=dilation, groups=groups, bias=bias),
            nn.InstanceNorm3d(out_planes, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=dilation, groups=groups, bias=bias),
            nn.InstanceNorm3d(out_planes, affine=False),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.FE(x)
        return x
class CoordConv_Block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, act_func='leaky', norm_layer='in', bias=True):
        super(CoordConv_Block, self).__init__()
        self.coordBlock_1 = nn.Sequential(
            nn.Conv3d(in_planes + 3, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True),
            nn.InstanceNorm3d(out_planes // 2, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True),
            nn.InstanceNorm3d(out_planes, affine=False),
        )
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x, coord):
        bias = self.coordBlock_1(torch.cat([x, coord], dim=1))
        return self.act(x + bias)


class BasicConv_Block_1D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, act_func='relu', norm_layer='bn', bias=False):
        super(BasicConv_Block_1D, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        if norm_layer == 'bn':
            self.bn = nn.BatchNorm1d(out_planes)
        elif norm_layer == 'in':
            self.bn = nn.InstanceNorm1d(out_planes, affine=False)
        elif norm_layer is None:
            self.bn = None
        else:
            assert False

        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func == 'leaky':
            self.act_func = nn.LeakyReLU()
        elif act_func == 'gelu':
            self.act_func = nn.GELU()
        elif act_func is None:
            self.act_func = None
        else:
            assert False

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act_func is not None:
            x = self.act_func(x)
        return x

def calcu_featureMap_dim(input_size, kernel, stride, padding, dilation):
    padding = np.tile(padding, len(input_size))
    kernel = np.tile(kernel, len(input_size))
    stride = np.tile(stride, len(input_size))
    dilation = np.tile(dilation, len(input_size))

    t_inputsize = np.array(input_size) + (padding * 2)
    t_kernel = (kernel-1) * dilation + 1
    output_size = (t_inputsize - t_kernel) // stride + 1
    return output_size


class AddCoords_size(nn.Module):
    def __init__(self, with_r=False):
        super(AddCoords_size, self).__init__()

        self.with_r = with_r
    def forward(self, size):
        # size = [xdim, y_dim, z_dim]
        x_dim = size[0]
        y_dim = size[1]
        z_dim = size[2]

        xx_ones = torch.ones([z_dim], dtype=torch.float32).cuda()
        xx_ones = xx_ones[:, None]  # (175, 1)
        xx_range = torch.arange(0, x_dim, dtype=torch.float32).cuda()  # (200,)
        xx_range = xx_range[None, :]  # (1, 200)
        xx_channel = torch.matmul(xx_ones, xx_range) # (175, 200)
        xx_channel = xx_channel.unsqueeze(-1).repeat(1, 1, y_dim).float() # (175, 200, 143)
        xx_channel /= (x_dim - 1)
        xx_channel = xx_channel * 2 - 1
        xx_channel = xx_channel.permute(1, 2, 0).unsqueeze(0)
        del xx_ones, xx_range

        yy_ones = torch.ones([x_dim], dtype=torch.float32).cuda()
        yy_ones = yy_ones[:, None]  # (batch, 175, 1)
        yy_range = torch.arange(0, y_dim, dtype=torch.float32).cuda()  # (200,)
        yy_range = yy_range[None, :]  # (batch, 1, 200)
        yy_channel = torch.matmul(yy_ones, yy_range) # (4, 175, 200)
        yy_channel = yy_channel.unsqueeze(-1).repeat(1, 1, z_dim).float() # (4, 1, 175, 200, 143)
        yy_channel /= (y_dim - 1)
        yy_channel = yy_channel * 2 - 1
        yy_channel = yy_channel.permute(0, 1, 2).unsqueeze(0)
        del yy_ones, yy_range

        zz_ones = torch.ones([y_dim], dtype=torch.float32).cuda()
        zz_ones = zz_ones[:, None]  # (batch, 175, 1)
        zz_range = torch.arange(0, z_dim, dtype=torch.float32).cuda()  # (200,)
        zz_range = zz_range[None, :]  # (batch, 1, 200)
        zz_channel = torch.matmul(zz_ones, zz_range) # (4, 175, 200)
        zz_channel = zz_channel.unsqueeze(-1).repeat(1, 1, x_dim).float() # (4, 1, 175, 200, 143)
        zz_channel /= (z_dim - 1)
        zz_channel = zz_channel * 2 - 1
        zz_channel = zz_channel.permute(2, 0, 1).unsqueeze(0)
        del zz_ones, zz_range

        ret = torch.cat([xx_channel, yy_channel, zz_channel], 0).unsqueeze(0)
        return ret

class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()

        self.with_r = with_r
    def forward(self, input_tensor):
        # batch, 1, x, y, z
        x_dim = input_tensor.shape[2]
        y_dim = input_tensor.shape[3]
        z_dim = input_tensor.shape[4]
        batch_size_tensor = input_tensor.shape[0]

        xx_ones = torch.ones([batch_size_tensor, z_dim], dtype=torch.float32).cuda()
        xx_ones = xx_ones[:, :, None]  # (batch, 175, 1)
        xx_range = torch.arange(0, x_dim, dtype=torch.float32).cuda()  # (200,)
        xx_range = xx_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, x_dim)  # (batch, 200)
        xx_range = xx_range[:, None, :]  # (batch, 1, 200)
        xx_channel = torch.matmul(xx_ones, xx_range) # (4, 175, 200)
        xx_channel = xx_channel.unsqueeze(3).repeat(1, 1, 1, y_dim).unsqueeze(1).float() # (4, 1, 175, 200, 143)
        del xx_ones, xx_range
        xx_channel /= (x_dim - 1)
        xx_channel = xx_channel * 2 - 1
        xx_channel = xx_channel.permute(0,1,3,4,2)

        yy_ones = torch.ones([batch_size_tensor, x_dim], dtype=torch.float32).cuda()
        yy_ones = yy_ones[:, :, None]  # (batch, 175, 1)
        yy_range = torch.arange(0, y_dim, dtype=torch.float32).cuda()  # (200,)
        yy_range = yy_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, y_dim)  # (batch, 200)
        yy_range = yy_range[:, None, :]  # (batch, 1, 200)
        yy_channel = torch.matmul(yy_ones, yy_range) # (4, 175, 200)
        yy_channel = yy_channel.unsqueeze(3).repeat(1, 1, 1, z_dim).unsqueeze(1).float() # (4, 1, 175, 200, 143)
        del yy_ones, yy_range
        yy_channel /= (y_dim - 1)
        yy_channel = yy_channel * 2 - 1
        yy_channel = yy_channel.permute(0, 1, 2, 3, 4)

        zz_ones = torch.ones([batch_size_tensor, y_dim], dtype=torch.float32).cuda()
        zz_ones = zz_ones[:, :, None]  # (batch, 175, 1)
        zz_range = torch.arange(0, z_dim, dtype=torch.float32).cuda()  # (200,)
        zz_range = zz_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, z_dim)  # (batch, 200)
        zz_range = zz_range[:, None, :]  # (batch, 1, 200)
        zz_channel = torch.matmul(zz_ones, zz_range) # (4, 175, 200)
        zz_channel = zz_channel.unsqueeze(3).repeat(1, 1, 1, x_dim).unsqueeze(1).float() # (4, 1, 175, 200, 143)
        del zz_ones, zz_range
        zz_channel /= (z_dim - 1)
        zz_channel = zz_channel * 2 - 1
        zz_channel = zz_channel.permute(0, 1, 4, 2, 3)
        ret = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], 1)
        # ret = torch.cat([xx_channel, yy_channel, zz_channel], 1)
        return ret

class CoordConv(nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1 , groups = 1, bias=False, with_r = False):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        self.conv = nn.Conv3d(in_channels+3, out_channels, kernel_size, stride, padding, dilation, groups=groups, bias=bias)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act_f = nn.LeakyReLU(inplace=True)

    def forward(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        ret = self.norm(ret)
        ret = self.act_f(ret)
        return ret



class sign_sqrt(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.sign(input) * torch.sqrt(torch.abs(input))
        # output = torch.sqrt(input.abs())
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]
        grad_input = torch.div(grad_output, ((torch.abs(output)+0.03)*2.))
        return grad_input

class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv3d, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Maxout(nn.Module):
    def __init__(self, d_in, d_out, pool_size):
        super(Maxout, self).__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m

class XceptionConv_layer(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(XceptionConv_layer, self).__init__()
        self.out_channels = out_planes
        self.conv = SeparableConv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Multi_Heads_Self_Attn_1D(nn.Module):
    def __init__(self, in_plane, out_plane, n_heads = 4, d_k = 16):
        super(Multi_Heads_Self_Attn_1D, self).__init__()
        self.num_heads = n_heads
        self.d_k = d_k

        """ query """
        self.query_conv = nn.Conv1d(in_plane, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=False)
        self.key_conv = nn.Conv1d(in_plane, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=False)
        self.value_conv = nn.Conv1d(in_plane, self.num_heads * self.d_k, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.output_conv = nn.Conv1d(self.num_heads * self.d_k, out_plane, kernel_size=1, bias=False)
        # self.drop_1 = nn.Dropout(p=0.1)
        self.act_f = nn.LeakyReLU(inplace=True)
        self.norm = nn.InstanceNorm1d(in_plane, affine=False)

        """ gamma """
        self.gamma = nn.Parameter(torch.zeros(1))
    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (x.permute(0, 2, 1, 3).contiguous()
                .view(shape[0], shape[2], shape[3] * self.num_heads))

    def forward(self, x):
        m_batchsize, C, depth= x.size()
        total_key_depth = depth

        """ linear for each component"""
        queries = self.query_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)
        keys = self.key_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)
        values = self.value_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)    # B X N X Channel

        """ split into multiple heads """
        queries = self._split_heads(queries) # B, head, N, d_k
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        """ query scale"""
        # query_scale = np.power(self.d_k, -0.5)
        # queries *= query_scale
        # logits = torch.matmul(queries, keys.permute(0,1,3,2))
        # weights = self.softmax(logits)  # BX (N) X (N/p)
        # out = torch.matmul(weights, values)

        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        weights = logits / self.d_k
        out = torch.matmul(weights, values)

        """ merge heads """
        out = self._merge_heads(out)

        """ linear to get output """
        out = out.view(m_batchsize, depth, -1).permute(0, 2, 1)
        out = self.output_conv(out)
        out = self.norm(out)
        # out = self.drop_1(out)
        out = out * self.gamma + x
        out = self.act_f(out)
        return out



class Multi_Heads_Self_Attn_pooling_3D(nn.Module):
    def __init__(self, in_plane, out_plane, n_heads = 4, d_k = 16):
        super(Multi_Heads_Self_Attn_pooling_3D, self).__init__()
        self.num_heads = n_heads
        self.d_k = d_k

        """ query """
        self.query_conv = nn.Conv3d(in_plane, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=True)
        self.key_conv = nn.Conv3d(in_plane, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=True)
        self.value_conv = nn.Conv3d(in_plane, self.num_heads * self.d_k, kernel_size=1, bias=True)
        self.drop = nn.Dropout(p=0.2)
        self.output_conv = nn.Conv3d(self.num_heads * self.d_k, out_plane, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.act_f = nn.LeakyReLU(inplace=True)
    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (x.permute(0, 2, 1, 3).contiguous()
                .view(shape[0], shape[2], shape[3] * self.num_heads))

    def forward(self, x):
        m_batchsize, C, width, height, depth= x.size()
        total_key_depth = width * height * depth

        """ linear for each component"""
        queries = self.query_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)
        keys = self.key_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)
        values = self.value_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)    # B X N X Channel

        """ split into multiple heads """
        queries = self._split_heads(queries) # B, head, N, d_k
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        """ readout """
        queries = queries.mean(-2, keepdim=True) # [batch, head, 1, feature]
        # queries = queries.mean(-2, keepdim=True) + queries.max(-2, keepdim=True)[0]  # [batch, head, 1, feature]

        """ query scale"""
        # query_scale = np.power(self.d_k, -0.5)
        # queries *= query_scale
        # logits = torch.matmul(queries, keys.permute(0,1,3,2)) # [batch, head, 1, length]
        # weights = self.drop(self.softmax(logits))  # BX (N) X (N/p)
        # out = torch.matmul(weights, values) # [batch, head, 1, length]

        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        weights =self.drop(logits / logits.size(-1))
        out = torch.matmul(weights, values) # batch, head, pooled, d_k

        """ merge heads """
        out = self._merge_heads(out) # batch, pooled, (head * d_k)

        """ linear to get output """
        out = out.view(m_batchsize, 1, 1, 1, -1).permute(0, 4, 1, 2, 3)
        # out = out.view(m_batchsize, 1, -1).permute(0, 2, 1) # (b, f, 1)
        out = self.output_conv(out)
        # out = self.norm(out)
        norm = nn.LayerNorm(out.size()[1:], elementwise_affine=False)
        out = norm(out)
        out = self.act_f(out)
        # out = self.drop_1(out)
        # out = out + x
        return out



class Multi_Heads_Self_Attn_pooling_1D(nn.Module):
    def __init__(self, in_plane, out_plane, n_heads = 4, d_k = 16):
        super(Multi_Heads_Self_Attn_pooling_1D, self).__init__()
        self.num_heads = n_heads
        self.d_k = d_k

        """ query """
        self.query_conv = nn.Conv1d(in_plane, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=False)
        self.key_conv = nn.Conv1d(in_plane, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=False)
        self.value_conv = nn.Conv1d(in_plane, self.num_heads * self.d_k, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.output_conv = nn.Conv1d(self.num_heads * self.d_k, out_plane, kernel_size=1, bias=False)
        # self.act_f = nn.LeakyReLU(inplace=True)
        # self.norm = nn.InstanceNorm1d(n_featuremap)
        # self.drop_1 = nn.Dropout(p=0.1)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (x.permute(0, 2, 1, 3).contiguous()
                .view(shape[0], shape[2], shape[3] * self.num_heads))

    def forward(self, x):
        m_batchsize, C, depth= x.size()
        total_key_depth = depth

        """ linear for each component"""
        queries = self.query_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)
        keys = self.key_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)
        values = self.value_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)    # B X N X Channel

        """ split into multiple heads """
        queries = self._split_heads(queries) # B, head, N, d_k
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        """ readout """
        # queries = queries.mean(-2, keepdim=True) # [batch, head, 1, feature]
        queries = queries.mean(-2, keepdim=True) + queries.max(-2, keepdim=True)[0]  # [batch, head, 1, feature]

        """ query scale"""
        # query_scale = np.power(self.d_k, -0.5)
        # p_queries *= query_scale
        # logits = torch.matmul(p_queries, keys.permute(0,1,3,2)) # [batch, head, 1, length]
        # weights = self.softmax(logits)  # BX (N) X (N/p)
        # out = torch.matmul(weights, values) # [batch, head, 1, length]

        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        weights = logits / self.d_k
        out = torch.matmul(weights, values)

        """ merge heads """
        out = self._merge_heads(out)

        """ linear to get output """
        out = out.view(m_batchsize, 1, -1).permute(0, 2, 1) # [b, f, 1]
        out = self.output_conv(out)
        # out = self.norm(out)
        # out = self.act_f(out)
        # out = self.drop_1(out)
        # out = out + x
        return out

class Multi_Heads_Self_Attn_pooling_1D_2(nn.Module):
    def __init__(self, n_featuremap, n_heads = 4,  d_k = 16):
        super(Multi_Heads_Self_Attn_pooling_1D_2, self).__init__()
        self.num_heads = n_heads
        self.d_k = d_k

        """ query """
        self.query_conv = nn.Conv1d(n_featuremap, self.num_heads * self.d_k , kernel_size=1, padding=0, bias=False)
        self.key_conv = nn.Conv1d(n_featuremap, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=False)
        self.value_conv = nn.Conv1d(n_featuremap, self.num_heads * self.d_k, kernel_size=1, bias=False)

        self.alpha = nn.Parameter(torch.zeros(self.num_heads, self.d_k))
        nn.init.normal_(self.alpha, mean=0.0, std=1.0)

        self.softmax = nn.Softmax(dim=-1)
        self.output_conv = nn.Conv1d(self.num_heads * self.d_k, n_featuremap, kernel_size=1, bias=False)
        self.act_f = nn.LeakyReLU(inplace=True)
        # self.norm = nn.InstanceNorm1d(n_featuremap)
        # self.drop_1 = nn.Dropout(p=0.1)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (x.permute(0, 2, 1, 3).contiguous()
                .view(shape[0], shape[2], shape[3] * self.num_heads))

    def forward(self, x):
        m_batchsize, C, depth= x.size()
        total_key_depth = depth

        """ linear for each component"""
        # queries = self.query_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)
        keys = self.key_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)
        values = self.value_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)    # B X N X Channel

        """ split into multiple heads """
        # queries = self._split_heads(queries) # B, head, N, d_k
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        """ readout """
        p_queries = self.alpha.unsqueeze(1).unsqueeze(0).repeat(x.size(0), 1, 1, 1)

        """ query scale"""
        # query_scale = np.power(self.d_k, -0.5)
        # p_queries *= query_scale
        # logits = torch.matmul(p_queries, keys.permute(0,1,3,2)) # [batch, head, 1, length]
        # weights = self.softmax(logits)  # BX (N) X (N/p)
        # out = torch.matmul(weights, values) # [batch, head, 1, length]

        logits = torch.matmul(p_queries, keys.permute(0, 1, 3, 2))
        weights = logits / self.d_k
        out = torch.matmul(weights, values)

        """ merge heads """
        out = self._merge_heads(out)

        """ linear to get output """
        out = out.view(m_batchsize, 1, -1).permute(0, 2, 1) # [b, f, 1]
        out = self.output_conv(out)
        # out = self.norm(out)
        # out = self.act_f(out)
        # out = self.drop_1(out)
        # out = out + x
        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, n_featuremap):
        super().__init__()
        self.ff_1 = nn.Conv1d(n_featuremap, n_featuremap // 2, kernel_size=1, bias=False)
        self.act_f = nn.LeakyReLU(inplace=True)
        self.ff_2 = nn.Conv1d(n_featuremap//2, n_featuremap, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm1d(n_featuremap)
        # self.drop_2 = nn.Dropout(p=0.1)

    def forward(self, x):
        residual = x
        out = self.ff_1(x)
        out = self.act_f(out)
        out = self.ff_2(out)
        out = self.norm(out)
        out = self.act_f(out)
        # out = self.drop_2(out)
        out = residual + out
        return out




class Multi_Heads_Self_Attn(nn.Module):
    def __init__(self, n_featuremap, n_heads = 4,  d_k = 16):
        super(Multi_Heads_Self_Attn, self).__init__()
        self.num_heads = n_heads
        self.d_k = d_k

        """ query """
        self.query_conv = nn.Conv3d(n_featuremap, self.num_heads * self.d_k , kernel_size=1, padding=0, bias=False)

        """ key """
        self.key_conv = nn.Conv3d(n_featuremap, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=False)

        """ value """
        self.value_conv = nn.Conv3d(n_featuremap, self.num_heads * self.d_k, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.output_conv = nn.Conv3d(self.num_heads * self.d_k, n_featuremap, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(n_featuremap)
        self.act_f = nn.LeakyReLU(inplace=True)

        # self.FF = nn.Conv3d(n_featuremap, n_featuremap, kernel_size=1, bias=False)
        # self.norm_FF = nn.InstanceNorm3d(n_featuremap)

        """ gamma """
        self.gamma = nn.Parameter(torch.zeros(1))

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (x.permute(0, 2, 1, 3).contiguous()
                .view(shape[0], shape[2], shape[3] * self.num_heads))

    def forward(self, x):
        m_batchsize, C, width, height, depth = x.size()
        total_key_depth = width * height * depth

        """ linear for each component"""
        queries = self.query_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)
        keys = self.key_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)
        values = self.value_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)    # B X N X Channel

        """ split into multiple heads """
        queries = self._split_heads(queries) # B, head, N, d_k
        keys = self._split_heads(keys )
        values = self._split_heads(values)

        """ query scale"""
        # query_scale = np.power(total_key_depth // self.num_heads, -0.5)
        # queries *= query_scale

        """Combine queries and keys """
        logits = torch.matmul(queries, keys.permute(0,1,3,2))
        # weights = self.softmax(logits)  # BX (N) X (N/p)
        weights = logits / total_key_depth

        out = torch.matmul(weights, values)

        """ merge heads """
        out = self._merge_heads(out)

        """ linear to get output """
        out = out.view(m_batchsize, width, height, depth, -1).permute(0, 4, 1, 2, 3)
        out = self.output_conv(out)
        out = self.norm(out)

        """ residual """
        # out = self.gamma * out
        out = out + x
        # out = torch.cat((out, x), 1)
        out = self.act_f(out)

        # """ FF """
        # residual = out
        # out = self.FF(out)
        # out = self.norm_FF(out)
        # out = out + residual
        # out = self.act_f(out)

        return out, weights, self.gamma

class Multi_Heads_Self_Attn_Q_KV(nn.Module):
    def __init__(self, n_featuremap_q, n_featuremap_kv, n_heads = 4, d_k = 16):
        super(Multi_Heads_Self_Attn_Q_KV, self).__init__()
        self.num_heads = n_heads
        self.d_k = d_k

        """ query """
        self.query_conv = nn.Conv3d(n_featuremap_q, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=False)

        """ key """
        self.key_conv = nn.Conv3d(n_featuremap_kv, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=False)

        """ value """
        self.value_conv = nn.Conv3d(n_featuremap_kv, self.num_heads * self.d_k, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.output_conv = nn.Conv3d(self.num_heads * self.d_k, n_featuremap_q, kernel_size=1, bias=False)
        # self.norm_layer = nn.BatchNorm3d(n_featuremap_kv)
        self.norm_layer = nn.InstanceNorm3d(n_featuremap_q)
        self.act_f = nn.LeakyReLU(inplace=True)

        """ gamma """
        self.gamma = nn.Parameter(torch.zeros(1))

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (x.permute(0, 2, 1, 3).contiguous()
                .view(shape[0], shape[2], shape[3] * self.num_heads))

    def forward(self, x_q, x_kv):
        m_batchsize, C, width, height, depth = x_q.size()
        total_q_depth = width * height * depth

        m_batchsize, C, width_kv, height_kv, depth_kv = x_kv.size()
        total_kv_depth = width_kv * height_kv * depth_kv

        """ linear for each component"""
        queries = self.query_conv(x_q).view(m_batchsize, -1, total_q_depth).permute(0, 2, 1)
        keys = self.key_conv(x_kv).view(m_batchsize, -1, total_kv_depth).permute(0, 2, 1)
        values = self.value_conv(x_kv).view(m_batchsize, -1, total_kv_depth).permute(0, 2, 1)    # B X N X Channel

        """ split into multiple heads """
        queries = self._split_heads(queries) # B, head, N, d_k
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        """ query scale"""
        # query_scale = np.power(total_kv_depth // self.num_heads, -0.5)
        # queries *= query_scale

        """Combine queries and keys """
        logits = torch.matmul(queries, keys.permute(0,1,3,2))
        # weights = self.softmax(logits)  # BX (N) X (N/p)
        weights = logits / total_kv_depth

        out = torch.matmul(weights, values)

        """ merge heads """
        out = self._merge_heads(out)

        """ linear to get output """
        out = out.view(m_batchsize, width, height, depth, -1).permute(0, 4, 1, 2, 3)
        out = self.output_conv(out)
        out = self.norm_layer(out)

        """ residual """
        # out = self.gamma * out
        out = out + x_q

        # x = (1-self.gamma) * x
        # out = torch.cat((out, x), 1)

        out = self.act_f(out)
        return out, weights, self.gamma

class Multi_Heads_Self_Attn_Q_KV_1D(nn.Module):
    def __init__(self, n_featuremap_q, n_featuremap_kv, n_heads = 4, d_k = 16):
        super(Multi_Heads_Self_Attn_Q_KV_1D, self).__init__()
        self.num_heads = n_heads
        self.d_k = d_k

        """ query """
        self.query_conv = nn.Conv1d(n_featuremap_q, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=False)

        """ key """
        self.key_conv = nn.Conv1d(n_featuremap_kv, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=False)

        """ value """
        self.value_conv = nn.Conv1d(n_featuremap_kv, self.num_heads * self.d_k, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.output_conv = nn.Conv1d(self.num_heads * self.d_k, n_featuremap_q, kernel_size=1, bias=False)

        # self.norm_layer = nn.InstanceNorm1d(n_featuremap_q)
        # self.act_f = nn.GELU()


        """ gamma """
        self.gamma = nn.Parameter(torch.zeros(1))
    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (x.permute(0, 2, 1, 3).contiguous()
                .view(shape[0], shape[2], shape[3] * self.num_heads))

    def forward(self, x_q, x_kv):
        m_batchsize, C, width = x_q.size()
        total_q_depth = width

        m_batchsize, C, width_kv = x_kv.size()
        total_kv_depth = width_kv

        """ linear for each component"""
        queries = self.query_conv(x_q).view(m_batchsize, -1, total_q_depth).permute(0, 2, 1)
        keys = self.key_conv(x_kv).view(m_batchsize, -1, total_kv_depth).permute(0, 2, 1)
        values = self.value_conv(x_kv).view(m_batchsize, -1, total_kv_depth).permute(0, 2, 1)    # B X N X Channel

        """ split into multiple heads """
        queries = self._split_heads(queries) # B, head, N, d_k
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        """ query scale"""
        # query_scale = np.power(self.d_k, -0.5)
        # queries *= query_scale
        # logits = torch.matmul(queries, keys.permute(0,1,3,2))
        # weights = self.softmax(logits)  # BX (N) X (N/p)

        """Combine queries and keys """
        logits = torch.matmul(queries, keys.permute(0,1,3,2))
        weights = logits / self.d_k
        out = torch.matmul(weights, values)

        """ merge heads """
        out = self._merge_heads(out)

        """ linear to get output """
        out = out.view(m_batchsize, width, -1).permute(0, 2, 1)
        out = self.output_conv(out)
        # out = self.norm_layer(out)

        """ residual """
        out = self.gamma * out
        out = out + x_q

        # x = (1-self.gamma) * x
        # out = torch.cat((out, x), 1)

        # out = self.act_f(out)
        return out


class gate_using_local_global(nn.Module):
    def __init__(self, n_local, n_global, n_plane):
        super(gate_using_local_global, self).__init__()

        """ query """
        self.local_conv = nn.Conv1d(n_local, n_plane, kernel_size=1, padding=0, bias=False)
        self.global_conv = nn.Conv1d(n_global, n_plane, kernel_size=1, padding=0, bias=False)

        self.gate = nn.Sequential(
            nn.Conv1d(n_plane * 2, n_plane // 2, kernel_size=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(n_plane // 2, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x_local, x_global):
        m_batchsize, C, width = x_local.size()
        total_q_depth = width

        m_batchsize, C, width_kv = x_global.size()
        total_kv_depth = width_kv

        """ linear for each component"""
        queries = self.local_conv(x_local).view(m_batchsize, -1, total_q_depth).permute(0, 2, 1)
        keys = self.global_conv(x_global).view(m_batchsize, -1, total_kv_depth).permute(0, 2, 1)
        values = self.value_conv(x_global).view(m_batchsize, -1, total_kv_depth).permute(0, 2, 1)    # B X N X Channel

        """ split into multiple heads """
        queries = self._split_heads(queries) # B, head, N, d_k
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        """ query scale"""
        # query_scale = np.power(self.d_k, -0.5)
        # queries *= query_scale
        # logits = torch.matmul(queries, keys.permute(0,1,3,2))
        # weights = self.softmax(logits)  # BX (N) X (N/p)

        """Combine queries and keys """
        logits = torch.matmul(queries, keys.permute(0,1,3,2))
        weights = logits / self.d_k
        out = torch.matmul(weights, values)

        """ merge heads """
        out = self._merge_heads(out)

        """ linear to get output """
        out = out.view(m_batchsize, width, -1).permute(0, 2, 1)
        out = self.output_conv(out)
        # out = self.norm_layer(out)

        """ residual """
        # out = self.gamma * out
        # out = out + x_q

        # x = (1-self.gamma) * x
        # out = torch.cat((out, x), 1)

        # out = self.act_f(out)
        return out


class VariationalPosterior(torch.nn.Module):
    def __init__(self, mu, rho):
        super(VariationalPosterior, self).__init__()
        self.mu = mu.cuda()
        self.rho = rho.cuda()
        # gaussian distribution to sample epsilon from
        self.normal = torch.distributions.Normal(0, 1)
        self.sigma = torch.log1p(torch.exp(self.rho)).cuda()

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).cuda()
        # reparametrizarion trick for sampling from posterior
        posterior_sample = (self.mu + self.sigma * epsilon).cuda()
        return posterior_sample

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()




class Prior(torch.nn.Module):
    '''
    Scaled Gaussian Mixtures for Priors
    '''
    def __init__(self, args):
        super(Prior, self).__init__()
        self.sig1 = args.sig1
        self.sig2 = args.sig2
        self.pi = args.pi


        self.s1 = torch.tensor([math.exp(-1. * self.sig1)], dtype=torch.float32).cuda()
        self.s2 = torch.tensor([math.exp(-1. * self.sig2)], dtype=torch.float32).cuda()

        self.gaussian1 = torch.distributions.Normal(0,self.s1)
        self.gaussian2 = torch.distributions.Normal(0,self.s2)


    def log_prob(self, input):
        input = input.cuda()
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1.-self.pi) * prob2)).sum()



class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class MS_GRU(nn.Module):
    def __init__(self, in_planes):
        super(MS_GRU, self).__init__()
        self.W_z = nn.Conv3d(in_planes * 2, in_planes, kernel_size=1, padding=0, bias=True)
        self.W_r = nn.Conv3d(in_planes * 2, in_planes, kernel_size=1, padding=0, bias=True)
        self.W_h = nn.Conv3d(in_planes * 2, in_planes, kernel_size=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, h_t_1, x_t):
        z_t = self.sigmoid(self.W_z(torch.cat([h_t_1, x_t], dim=1)))
        r_t = self.sigmoid(self.W_r(torch.cat([h_t_1, x_t], dim=1)))
        h_t_til = self.tanh(self.W_h(torch.cat([r_t * h_t_1, x_t], dim=1)))
        h_t = (1 - z_t) * h_t_1 + z_t * h_t_til
        return h_t

class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input
