# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:35:21 2020

@author: ZJU
"""
import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x))
import torch
import numpy as np
from PIL import Image

class injective_pad(nn.Module):
    def __init__(self, pad_size):
        super().__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :x.size(1) - self.pad_size, :, :]

class cWCT(torch.nn.Module):
    def __init__(self, eps=2e-5, use_double=False):
        super().__init__()
        self.eps = eps
        self.use_double = use_double

    def transfer(self, cont_feat, styl_feat, cmask=None, smask=None):
        if cmask is None or smask is None:
            return self._transfer(cont_feat, styl_feat)
        else:
            return self._transfer_seg(cont_feat, styl_feat, cmask, smask)

    def _transfer(self, cont_feat, styl_feat):
        """
        :param cont_feat: [B, N, cH, cW]
        :param styl_feat: [B, N, sH, sW]
        :return color_fea: [B, N, cH, cW]
        """
        B, N, cH, cW = cont_feat.shape
        cont_feat = cont_feat.reshape(B, N, -1)
        styl_feat = styl_feat.reshape(B, N, -1)

        in_dtype = cont_feat.dtype
        if self.use_double:
            cont_feat = cont_feat.double()
            styl_feat = styl_feat.double()

        # whitening and coloring transforms
        whiten_fea = self.whitening(cont_feat)
        color_fea = self.coloring(whiten_fea, styl_feat)

        if self.use_double:
            color_fea = color_fea.to(in_dtype)

        return color_fea.reshape(B, N, cH, cW)

    def _transfer_seg(self, cont_feat, styl_feat, cmask, smask):
        """
        :param cont_feat: [B, N, cH, cW]
        :param styl_feat: [B, N, sH, sW]
        :param cmask: numpy [B, _, _]
        :param smask: numpy [B, _, _]
        :return color_fea: [B, N, cH, cW]
        """
        B, N, cH, cW = cont_feat.shape
        _, _, sH, sW = styl_feat.shape
        cont_feat = cont_feat.reshape(B, N, -1)
        styl_feat = styl_feat.reshape(B, N, -1)

        in_dtype = cont_feat.dtype
        if self.use_double:
            cont_feat = cont_feat.double()
            styl_feat = styl_feat.double()

        for i in range(B):
            label_set, label_indicator = self.compute_label_info(cmask[i], smask[i])
            resized_content_segment = self.resize(cmask[i], cH, cW)
            resized_style_segment = self.resize(smask[i], sH, sW)

            single_content_feat = cont_feat[i]     # [N, cH*cW]
            single_style_feat = styl_feat[i]   # [N, sH*sW]
            target_feature = single_content_feat.clone()   # [N, cH*cW]

            for label in label_set:
                if not label_indicator[label]:
                    continue

                content_index = self.get_index(resized_content_segment, label).to(single_content_feat.device)
                style_index = self.get_index(resized_style_segment, label).to(single_style_feat.device)
                if content_index is None or style_index is None:
                    continue

                masked_content_feat = torch.index_select(single_content_feat, 1, content_index)
                masked_style_feat = torch.index_select(single_style_feat, 1, style_index)
                whiten_fea = self.whitening(masked_content_feat)
                _target_feature = self.coloring(whiten_fea, masked_style_feat)

                new_target_feature = torch.transpose(target_feature, 1, 0)
                new_target_feature.index_copy_(0, content_index,
                                               torch.transpose(_target_feature, 1, 0))
                target_feature = torch.transpose(new_target_feature, 1, 0)

            cont_feat[i] = target_feature
        color_fea = cont_feat

        if self.use_double:
            color_fea = color_fea.to(in_dtype)

        return color_fea.reshape(B, N, cH, cW)

    def cholesky_dec(self, conv, invert=False):
        cholesky = torch.linalg.cholesky if torch.__version__ >= '1.8.0' else torch.cholesky
        try:
            L = cholesky(conv)
        except RuntimeError:
            # print("Warning: Cholesky Decomposition fails")
            iden = torch.eye(conv.shape[-1]).to(conv.device)
            eps = self.eps
            while True:
                try:
                    conv = conv + iden * eps
                    L = cholesky(conv)
                    break
                except RuntimeError:
                    eps = eps+self.eps

        if invert:
            L = torch.inverse(L)

        return L.to(conv.dtype)

    def whitening(self, x):
        mean = torch.mean(x, -1)
        mean = mean.unsqueeze(-1).expand_as(x)
        x = x - mean

        conv = (x @ x.transpose(-1, -2)).div(x.shape[-1] - 1)
        inv_L = self.cholesky_dec(conv, invert=True)

        whiten_x = inv_L @ x

        return whiten_x

    def coloring(self, whiten_xc, xs):
        xs_mean = torch.mean(xs, -1)
        xs = xs - xs_mean.unsqueeze(-1).expand_as(xs)

        conv = (xs @ xs.transpose(-1, -2)).div(xs.shape[-1] - 1)
        Ls = self.cholesky_dec(conv, invert=False)

        coloring_cs = Ls @ whiten_xc
        coloring_cs = coloring_cs + xs_mean.unsqueeze(-1).expand_as(coloring_cs)

        return coloring_cs

    def compute_label_info(self, cont_seg, styl_seg):
        if cont_seg.size is False or styl_seg.size is False:
            return
        max_label = np.max(cont_seg) + 1
        self.label_set = np.unique(cont_seg)
        self.label_indicator = np.zeros(max_label)
        for l in self.label_set:
            is_valid = lambda a, b: a > 10 and b > 10 and a / b < 100 and b / a < 100
            o_cont_mask = np.where(cont_seg.reshape(cont_seg.shape[0] * cont_seg.shape[1]) == l)
            o_styl_mask = np.where(styl_seg.reshape(styl_seg.shape[0] * styl_seg.shape[1]) == l)
            self.label_indicator[l] = is_valid(o_cont_mask[0].size, o_styl_mask[0].size)
        return self.label_set, self.label_indicator

    def resize(self, img, H, W):
        size = (W, H)
        if len(img.shape) == 2:
            return np.array(Image.fromarray(img).resize(size, Image.NEAREST))
        else:
            return np.array(Image.fromarray(img, mode='RGB').resize(size, Image.NEAREST))

    def get_index(self, feat, label):
        mask = np.where(feat.reshape(feat.shape[0] * feat.shape[1]) == label)
        if mask[0].size <= 0:
            return None
        return torch.LongTensor(mask[0])

    def interpolation(self, cont_feat, styl_feat_list, alpha_s_list, alpha_c=0.0):
        """
        :param cont_feat: Tensor [B, N, cH, cW]
        :param styl_feat_list: List [Tensor [B, N, _, _], Tensor [B, N, _, _], ...]
        :param alpha_s_list: List [float, float, ...]
        :param alpha_c: float
        :return color_fea: Tensor [B, N, cH, cW]
        """
        assert len(styl_feat_list) == len(alpha_s_list)

        B, N, cH, cW = cont_feat.shape
        cont_feat = cont_feat.reshape(B, N, -1)

        in_dtype = cont_feat.dtype
        if self.use_double:
            cont_feat = cont_feat.double()

        c_mean = torch.mean(cont_feat, -1)
        cont_feat = cont_feat - c_mean.unsqueeze(-1).expand_as(cont_feat)

        cont_conv = (cont_feat @ cont_feat.transpose(-1, -2)).div(cont_feat.shape[-1] - 1)  # interpolate Conv works well
        inv_Lc = self.cholesky_dec(cont_conv, invert=True)  # interpolate L seems to be slightly better

        whiten_c = inv_Lc @ cont_feat

        # First interpolate between style_A, style_B, style_C, ...
        mix_Ls = torch.zeros_like(inv_Lc)   # [B, N, N]
        mix_s_mean = torch.zeros_like(c_mean)   # [B, N]
        for styl_feat, alpha_s in zip(styl_feat_list, alpha_s_list):
            assert styl_feat.shape[0] == B and styl_feat.shape[1] == N
            styl_feat = styl_feat.reshape(B, N, -1)

            if self.use_double:
                styl_feat = styl_feat.double()

            s_mean = torch.mean(styl_feat, -1)
            styl_feat = styl_feat - s_mean.unsqueeze(-1).expand_as(styl_feat)

            styl_conv = (styl_feat @ styl_feat.transpose(-1, -2)).div(styl_feat.shape[-1] - 1)  # interpolate Conv works well
            Ls = self.cholesky_dec(styl_conv, invert=False)  # interpolate L seems to be slightly better

            mix_Ls += Ls * alpha_s
            mix_s_mean += s_mean * alpha_s

        # Second interpolate between content and style_mix
        if alpha_c != 0.0:
            Lc = self.cholesky_dec(cont_conv, invert=False)
            mix_Ls = mix_Ls * (1-alpha_c) + Lc * alpha_c
            mix_s_mean = mix_s_mean * (1-alpha_c) + c_mean * alpha_c

        color_fea = mix_Ls @ whiten_c
        color_fea = color_fea + mix_s_mean.unsqueeze(-1).expand_as(color_fea)

        if self.use_double:
            color_fea = color_fea.to(in_dtype)

        return color_fea.reshape(B, N, cH, cW)


if __name__ == '__main__':
    # transfer
    c = torch.rand((2, 16, 512, 256))
    s = torch.rand((2, 16, 64, 128))

    cwct = cWCT(use_double=True)
    cs = cwct.transfer(c, s)
    print(cs.shape)


    # interpolation
    c = torch.rand((1, 16, 512, 256))
    s_list = [torch.rand((1, 16, 64, 128)) for _ in range(4)]
    alpha_s_list = [0.25 for _ in range(4)]     # interpolate between style_A, style_B, style_C, ...
    alpha_c = 0.5   # interpolate between content and style_mix if alpha_c!=0.0

    cwct = cWCT(use_double=True)
    cs = cwct.interpolation(c, s_list, alpha_s_list, alpha_c)
    print(cs.shape)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std

def mean_normalization(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size))/std.expand(size)
    return normalized_feat
    

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, content, style):
       
        assert (content.size()[:2] == style.size()[:2])
        size = content.size()
        style_mean, style_std = calc_mean_std(style)
        content_mean, content_std = calc_mean_std(content)
        normalized_feat = (content - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class Attention(nn.Module):
    def __init__(self, num_features):
        super(Attention, self).__init__()
        self.query_conv = nn.Conv2d(num_features, num_features, (1, 1))
        self.key_conv = nn.Conv2d(num_features, num_features, (1, 1))
        self.value_conv = nn.Conv2d(num_features, num_features, (1, 1))
        self.softmax = nn.Softmax(dim = -1)
        nn.init.xavier_uniform_(self.query_conv.weight)
        nn.init.uniform_(self.query_conv.bias, 0.0, 1.0)
        nn.init.xavier_uniform_(self.key_conv.weight)
        nn.init.uniform_(self.key_conv.bias, 0.0, 1.0)
        nn.init.xavier_uniform_(self.value_conv.weight)
        nn.init.uniform_(self.value_conv.bias, 0.0, 1.0)
        
    def forward(self, content_feat, style_feat):
        Query = self.query_conv(mean_normalization(content_feat))
        Key = self.key_conv(mean_normalization(style_feat))
        Value = self.value_conv(style_feat)
        batch_size, channels, height_c, width_c = Query.size()
        Query = Query.view(batch_size, -1, width_c * height_c).permute(0, 2, 1)
        batch_size, channels, height_s, width_s = Key.size()
        Key = Key.view(batch_size, -1, width_s * height_s)
        Attention_Weights = self.softmax(torch.bmm(Query, Key))

        Value = Value.view(batch_size, -1, width_s * height_s)
        Output = torch.bmm(Value, Attention_Weights.permute(0, 2, 1))
        Output = Output.view(batch_size, channels, height_c, width_c)
        return Output


class SAFIN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        #self.shared_weight = nn.Parameter(torch.Tensor(num_features), requires_grad=True)
        #self.shared_bias = nn.Parameter(torch.Tensor(num_features), requires_grad=True)
        #self.shared_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.gamma_conv = nn.Conv2d(num_features, num_features, (1, 1))
        self.beta_conv = nn.Conv2d(num_features, num_features, (1, 1))
        self.attention = Attention(num_features)
        self.relu = nn.ReLU()
       # nn.init.ones_(self.shared_weight)
        #nn.init.zeros_(self.shared_bias)
        nn.init.xavier_uniform_(self.gamma_conv.weight)
        nn.init.uniform_(self.gamma_conv.bias, 0.0, 1.0)
        nn.init.xavier_uniform_(self.beta_conv.weight)
        nn.init.uniform_(self.beta_conv.bias, 0.0, 1.0)

    def forward(self, content_feat, style_feat, output_shared=False):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_feat = self.attention(content_feat, style_feat)
        style_gamma = self.relu(self.gamma_conv(style_feat))
        style_beta = self.relu(self.beta_conv(style_feat))
        content_mean, content_std = calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        #shared_affine_feat = normalized_feat * self.shared_weight.view(1, self.num_features, 1, 1).expand(size) + \
                             #self.shared_bias.view(1, self.num_features, 1, 1).expand(size)
        #if output_shared:
        #    return shared_affine_feat
        output = normalized_feat * style_gamma + style_beta
        return output
    
class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
            
        return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class flazy2d(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super().__init__()
        
        if out_channel is None:
            out_channel = in_channel
        weight = torch.randn(in_channel, out_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape
        out = F.conv2d(input, self.weight)
        return out

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super().__init__()

        if out_channel is None:
            out_channel = in_channel
        weight = np.random.randn(in_channel, out_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape
        weight = self.calc_weight()
        out = F.conv2d(input, weight)
        return out

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
   
    def forward(self, x):
        
        x = self.conv(x)
        x=self.bn(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 2, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)*x

class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialGate, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)*x

        
class _SpatialGate(nn.Module):
    def __init__(self,in_channel):
        super(_SpatialGate, self).__init__()
        self.spatial = CBAM(in_channel)
        
    def _forward(self, x):    
        return self.spatial(x)
      
    def forward(self,input):
        in_a, in_b = input.chunk(2, 1)
        netout=self._forward(in_a)
        out_b=in_b+netout
        
        return torch.cat([in_a, out_b], 1)
    def reverse(self,output):
        out_a, out_b = output.chunk(2, 1)
        netout= self._forward(out_a)
        in_b=out_b-netout
        
        return torch.cat([out_a, in_b], 1)

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine
        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )
        self.att=CBAM(in_channel//2)
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()
        self.w = nn.Parameter(torch.ones(2))
    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)
    
     
        
        if self.affine:
           
            log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

        else:
            w0=torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
            w1=torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
          
            net_out = w0*self.net(in_a)+w1*self.att(in_a)
            out_b = in_b + net_out
      
        return torch.cat([out_b, in_a], 1)

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)
       
        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            w0=torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
            w1=torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
            net_out = w0*self.net(out_a)+w1*self.att(out_a)
            in_b = out_b - net_out
        
        
        return torch.cat([in_b, out_a], 1)



class Flow(nn.Module):
    def __init__(self, in_channel, use_coupling=True, affine=True, conv_lu=True):
        super().__init__()

        #self.actnorm = ActNorm(in_channel)
       
        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)
            
        self.use_coupling = use_coupling
        if self.use_coupling:
            self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        
       # input = self.actnorm(input)
        input = self.invconv(input)
        if self.use_coupling:
            input = self.coupling(input)
       # input =self.att(input)
        return input

    def reverse(self, input):
       # input =self.att.reverse(input)
        if self.use_coupling:
            input = self.coupling.reverse(input)
      
        input = self.invconv.reverse(input)
        #input = self.actnorm.reverse(input)
        #input =self.att.reverse(input)
        #input=self.invconv.reverse(input)
        return input



def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

    def forward(self, input):
       
        b_size, n_channel, height, width = input.shape
        
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
        for flow in self.flows:
            out = flow(out)
     
        return out
      

    def reverse(self, output, reconstruct=False):
        input = output
        for flow in self.flows[::-1]:
            input = flow.reverse(input)
           
        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )
       
        return unsqueezed

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(inplace=True),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(inplace=True),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(inplace=True),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(inplace=True),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True)  # relu5-4
)

projection_style = nn.Sequential(
    nn.Linear(in_features=256, out_features=128),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=128, out_features=128)
)

projection_content = nn.Sequential(
    nn.Linear(in_features=512, out_features=256),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=256, out_features=128)
)

project_1=nn.Sequential(
    nn.Linear(in_features=128, out_features=128),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=128, out_features=128)
)



class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs




class Net(nn.Module):
    def __init__(self,encoder,in_channel, n_flow, n_block, affine=True, conv_lu=True):
        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4','enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block -1 ):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 4
        #self.decoder=decoder
       # self.embedding=Embed
        self.blocks.append(Block(n_channel, n_flow, affine=affine))
        self.wct = cWCT()
        #projection
        self.proj_style = projection_style
        self.proj_content = projection_content
        self.project_1=project_1
        self.batch_size=4
        self.pad = 5
        self.inj_pad = injective_pad(self.pad)
        #transform
        #self.transform = Transform(in_planes = 512)
        self.w = nn.Parameter(torch.ones(4))
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.bn = nn.BatchNorm1d(128, affine=False)

          
        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target, norm = False):
        if(norm == False):
          return self.mse_loss(input, target)
        else:
          return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def compute_contrastive_loss(self, feat_q, feat_k, tau, index):
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / tau
        #loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
       
        
        loss = self.cross_entropy_loss(out, torch.tensor([index], dtype=torch.long, device=feat_q.device))
       
        return loss

    def style_feature_contrastive(self, input):
        # out = self.enc_style(input)
        input = torch.sum(input, dim=[2, 3])
        size=input.size()
        if size[1]==256:
          input = self.proj_style(input)
        elif size[1]==128:
          input= self.project_1(input)
        else:
          input=self.proj_content(input)
        #input = input / torch.norm(input, p=2, dim=1, keepdim=True)
        return input

    def content_feature_contrastive(self, input):
        #out = self.enc_content(input)
        out = torch.sum(input, dim=[2, 3])
        out = self.proj_content(out)
       # out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return out
    
        
    def forward(self, input, forward=True, style=None):
        if forward:
            return self._forward(input, style=style)
        else:
            return self._reverse(input, style=style)

    def _forward(self, input, style=None):
        
        for block in self.blocks:
            input = block(input)
        if style is not None:
            input = self.adain(input, style)
        return input



    def weighted_mse_mean(self,input,target,weights = None):
      assert input.size() == target.size()
      size = input.size()
      if weights == None:
        weights = torch.ones(size = size[0])
      
      if len(size) == 4: # gram matrix is B,C,C
      
        se = ((input.view(size[0],-1) - target.view(size[0],-1))**2)
        
        return (se.mean(dim = 1)*weights).mean()
        
    
    
    def weighted_mse_loss(self,input,target,weights = None):
      assert input.size() == target.size()
      size = input.size()
      if weights == None:
        weights = torch.ones(size = size[0])
        
      if len(size) == 3: # gram matrix is B,C,C
        se = ((input.view(size[0],-1) - target.view(size[0],-1))**2)
       
        return (se.mean(dim = 1)*weights).mean()
        
    def gram_matrix(self,x, normalize=True):
        '''
        Generate gram matrices of the representations of content and style images.
        '''
        (b, ch, h, w) = x.size()
        features = x.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t)
        if normalize:
            gram /= ch * h * w
        return gram
    
    def calc_ast_style_loss_normalized(self, input, target): # this replaces the style loss for all AST models in the training
            G1 = self.gram_matrix(input,False)
            G2 = self.gram_matrix(target,False).detach() # we dont need the gradient of the target
    
            size = input.size()
            assert(len(size) == 4)
    
            g1_norm = torch.linalg.norm(G1,dim = (1,2))
            g2_norm = torch.linalg.norm(G2,dim = (1,2))
    
            size = G1.size()
            Nl = size[1] * size[2] # Or C x C = C^2
            normalize_term =  (torch.square(g1_norm) + torch.square(g2_norm))/Nl  #
    
            weights = (1/normalize_term)
            #weights = weights.view(size[0],1,1)
            return self.weighted_mse_loss(G1,G2,weights)
    def calc_per_style_loss_normalized(self, input, target): # this replaces the style loss for all AST models in the training
            input_mean, input_std = calc_mean_std(input)
            target_mean, target_std = calc_mean_std(target)
            size = input.size()
            assert(len(size) == 4)
            
            input_mean2= torch.linalg.norm(input_mean.view(size[0],-1),dim = (1))
            target_mean2= torch.linalg.norm(target_mean.view(size[0],-1),dim = (1))
            input_std2= torch.linalg.norm(input_std.view(size[0],-1),dim = (1))
            target_std2= torch.linalg.norm(target_std.view(size[0],-1),dim = (1))
    
           # size = G1.size()
            #Nl = size[1] * size[2] # Or C x C = C^2
            nl=size[1]
            normalize_mean =  (torch.square(input_mean2) + torch.square(target_mean2)) /nl #
            normalize_std =  (torch.square(input_std2) + torch.square(target_std2))/nl
            
            #weights = weights.view(size[0],1,1)
            return self.weighted_mse_mean(input_mean,target_mean,1/normalize_mean)+self.weighted_mse_mean(input_std,target_std,1/normalize_std)

    def off_diagonal(self,x):
    # return a flattened view of the off-diagonal elements of a square matrix
      n, m = x.shape
      assert n == m
      return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _reverse(self, content, style=None):
      
      
       
        
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        content=self.inj_pad.forward(content)
        style=self.inj_pad.forward(style)
        content=self._forward(content, style=None)
        style=self._forward(style, style=None)
       
       
        # postional embedding is calculated in transformer.py
      
        out = self.wct.transfer(content,style)
       
        for i, block in enumerate(self.blocks[::-1]):
            out = block.reverse(out)
        out=self.inj_pad.inverse(out)
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        w4 = torch.exp(self.w[3]) / torch.sum(torch.exp(self.w))
       
        loss=nn.Parameter(torch.ones(5),requires_grad=False)
        #g_t = self.decoder(stylized)
        g_t_feats = self.encode_with_intermediate(out)
        for i in range(1,5):
          c = self.bn(self.style_feature_contrastive(g_t_feats[i])).T @ self.bn(self.style_feature_contrastive(style_feats[i]))
          c.div_(self.batch_size)
          on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
      
          off_diag = self.off_diagonal(c).pow_(2).sum()
          loss[i] = on_diag + 0.0051 * off_diag
        loss_sup=w1*loss[1]+w2*loss[2]+w3*loss[3]+w4*loss[4]
        loss_c = self.calc_content_loss(g_t_feats[4], content_feats[4], norm = True)
        loss_s = self.calc_ast_style_loss_normalized(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_ast_style_loss_normalized(g_t_feats[i], style_feats[i])
        #c = self.bn(self.content_feature_contrastive(g_t_feats[3])).T @ self.bn(self.content_feature_contrastive(content_feats[3]))
        #c.div_(self.batch_size)
        #on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        #off_diag = self.off_diagonal(c).pow_(2).sum()
        #loss_supc = on_diag + 0.0051 * off_diag
        """IDENTITY LOSSES"""
        #Icc =self.adain(content,content)
        #for i, block in enumerate(self.blocks[::-1]):
        #    Icc = block.reverse(Icc)
        #Iss =self.adain(style,style)
        #for i, block in enumerate(self.blocks[::-1]):
        #    Iss = block.reverse(Iss)
        #l_identity1 = self.calc_content_loss(Icc, content_images) + self.calc_content_loss(Iss, style_images)
       # Fcc = self.encode_with_intermediate(Icc)
       # Fss = self.encode_with_intermediate(Iss)
       # l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
       # for i in range(1, 4):
       #     l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i], style_feats[i])
       
      
        
        # Contrastive learning.
        '''
        half = int(batch_size / 2)
        style_up = self.style_feature_contrastive(g_t_feats[2][0:half])
        style_down = self.style_feature_contrastive(g_t_feats[2][half:])
        content_up = self.content_feature_contrastive(g_t_feats[3][0:half])
        content_down = self.content_feature_contrastive(g_t_feats[3][half:])
       
        style_contrastive_loss = 0
       
        for i in range(half):
            reference_style = style_up[i:i+1]

            if i ==0:
                style_comparisons = torch.cat([style_down[0:half-1], style_up[1:]], 0)
            elif i == 1:
                style_comparisons = torch.cat([style_down[1:], style_up[0:1], style_up[2:]], 0)
            elif i == (half-1):
                style_comparisons = torch.cat([style_down[half-1:], style_down[0:half-2], style_up[0:half-1]], 0)
            else:
                style_comparisons = torch.cat([style_down[i:], style_down[0:i-1], style_up[0:i], style_up[i+1:]], 0)

            style_contrastive_loss += self.compute_contrastive_loss(reference_style, style_comparisons, 0.2, 0)

        for i in range(half):
            reference_style = style_down[i:i+1]

            if i ==0:
                style_comparisons = torch.cat([style_up[0:1], style_up[2:], style_down[1:]], 0)
            elif i == (half-2):
                style_comparisons = torch.cat([style_up[half-2:half-1], style_up[0:half-2], style_down[0:half-2], style_down[half-1:]], 0)
            elif i == (half-1):
                style_comparisons = torch.cat([style_up[half-1:], style_up[1:half-1], style_down[0:half-1]], 0)
            else:
                style_comparisons = torch.cat([style_up[i:i+1], style_up[0:i], style_up[i+2:], style_down[0:i], style_down[i+1:]], 0)

            style_contrastive_loss += self.compute_contrastive_loss(reference_style, style_comparisons, 0.2, 0)


        content_contrastive_loss = 0
        for i in range(half):
            reference_content = content_up[i:i+1]

            if i == 0:
                content_comparisons = torch.cat([content_down[half-1:], content_down[1:half-1], content_up[1:]], 0)
            elif i == 1:
                content_comparisons = torch.cat([content_down[0:1], content_down[2:], content_up[0:1], content_up[2:]], 0)
            elif i == (half-1):
                content_comparisons = torch.cat([content_down[half-2:half-1], content_down[0:half-2], content_up[0:half-1]], 0)
            else:
                content_comparisons = torch.cat([content_down[i-1:i], content_down[0:i-1], content_down[i+1:], content_up[0:i], content_up[i+1:]], 0)

            content_contrastive_loss += self.compute_contrastive_loss(reference_content, content_comparisons, 0.2, 0)

        for i in range(half):
            reference_content = content_down[i:i+1]

            if i == 0:
                content_comparisons = torch.cat([content_up[1:], content_down[1:]], 0)
            elif i == (half-2):
                content_comparisons = torch.cat([content_up[half-1:], content_up[0:half-2], content_down[0:half-2], content_down[half-1:]], 0)
            elif i == (half-1):
                content_comparisons = torch.cat([content_up[0:half-1], content_down[0:half-1]], 0)
            else:
                content_comparisons = torch.cat([content_up[i+1:i+2], content_up[0:i], content_up[i+2:], content_down[0:i], content_down[i+1:]], 0)

            content_contrastive_loss += self.compute_contrastive_loss(reference_content, content_comparisons, 0.2, 0)
      
       # return out, loss_c, loss_s, content_contrastive_loss, style_contrastive_loss
        '''
       
        return out, loss_c, loss_s,loss_sup
        return out