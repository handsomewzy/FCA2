import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FF
import colorsys
import math
from mmcv.cnn import ConvModule
from mmengine import MMLogger, print_log
from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint
from basicsr.utils.registry import ARCH_REGISTRY
from tqdm import tqdm
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        normal_init, update_init_info,
                                        xavier_init)
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from torch import Tensor
# from .sr_backbone import default_init_weights

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
                # m.append(nn.GroupNorm(n_feats//4,n_feats))
            if i == 0:
                m.append(act)
                # m.append(quantize.Quantization(bit=2, qq_bit=8, finetune=False))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels: int = 64, res_scale: float = 1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style modules.
        For modules with residual paths, using smaller std is better for
        stability and performance. We empirically use 0.1. See more details in
        "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Flow_BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_blocks=30):
        super(Flow_BasicBlock, self).__init__()
        main = []
        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))
        self.main = nn.Sequential(*main)
        
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_channels != out_channels:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(out_channels)
        #     )

    # def forward(self, x):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = self.bn2(self.conv2(out))
    #     out += self.shortcut(x)
    #     out = F.relu(out)
        
        
    #     return out

    def forward(self, feat):
        """Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)


def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    if dilation == 1:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), bias=bias)
    elif dilation == 2:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=2, bias=bias, dilation=dilation)

    else:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=3, bias=bias, dilation=dilation)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResAttentionBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttentionBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
                # m.append(nn.GroupNorm(n_feats//4,n_feats))
            if i == 0:
                m.append(act)
                # m.append(quantize.Quantization(bit=2, qq_bit=8, finetune=False))
        m.append(CALayer(n_feats, 3))
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class SSB(nn.Module):
    def __init__(self, n_feats, kernel_size, act, res_scale, conv=default_conv):
        super(SSB, self).__init__()
        self.spa = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.spc = ResAttentionBlock(conv, n_feats, 1, act=act, res_scale=res_scale)
      
    def forward(self, x):
        return self.spc(self.spa(x))


class SSPN(nn.Module):
    def __init__(self, n_feats, n_blocks, act, res_scale):
        super(SSPN, self).__init__()
        kernel_size = 3
        m = []
        for i in range(n_blocks):
            m.append(SSB(n_feats, kernel_size, act=act, res_scale=res_scale))
        self.net = nn.Sequential(*m)

    def forward(self, x):
        res = self.net(x)
        res = res + x
        return res


# a single branch of proposed SSPSR
class BranchUnit(nn.Module):
    def __init__(self, n_colors, n_feats, n_blocks, act, res_scale, up_scale, use_tail=True, conv=default_conv):
        super(BranchUnit, self).__init__()
        kernel_size = 3
        # self.head = conv(n_colors, n_feats, kernel_size)
        self.head = nn.Conv2d(n_colors, n_feats, kernel_size=3, padding=1)
        self.body = SSPN(n_feats, n_blocks, act, res_scale)
        self.upsample = Upsampler(conv, up_scale, n_feats)
        self.tail = None
        if use_tail:
            self.tail = conv(n_feats, n_colors, kernel_size)

    def forward(self, x):
        y = self.head(x)
        y = self.body(y)
        y = self.upsample(y)
        if self.tail is not None:
            y = self.tail(y)
        return y
    
    
class Encoder(nn.Module):
    def __init__(self, input_channel, out_channel,n_feats=128):
        super(Encoder, self).__init__()
        self.input_channel = input_channel
        self.out_channel = out_channel
        self.branch = BranchUnit(input_channel, n_feats=n_feats, n_blocks=3, act=nn.LeakyReLU(), res_scale=0.1,use_tail=False, up_scale=1,conv=default_conv)
        self.final = nn.Conv2d(n_feats, out_channel, kernel_size=3, padding=1) 

    def forward(self, x):
        x = self.branch(x)
        x = self.final(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_channel, out_channel, n_feats=128):
        super(Decoder, self).__init__()
        self.input_channel = input_channel
        self.out_channel = out_channel
        self.branch = BranchUnit(input_channel, n_feats=n_feats, n_blocks=3, act=nn.LeakyReLU(), res_scale=0.1, up_scale=1,conv=default_conv,use_tail=False)
        self.final = nn.Conv2d(n_feats, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.branch(x)
        x = self.final(x)
        return x

@ARCH_REGISTRY.register()
class GAE(nn.Module):
    def __init__(self, n_subs=9, n_ovls=3, frame=7, n_feats=64, spynet_pretrained=None):
        """
        Initializes the GAE (Group-Autoencoder) network.
        
        Args:
            n_subs (int): Number of spectral subsets to process at each stage.
            n_ovls (int): Number of overlapping spectral channels between subsets.
            frame (int): Number of frames in the input data, multiplied by 3 to match RGB channels.
            n_feats (int): Number of features for internal representation in the encoder and decoder.
        """
        super(GAE, self).__init__()
        self.frame = frame
        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)
        
        # Initialize the encoder and decoder
        self.Encoder = Encoder(n_subs, 3, n_feats)
        self.Decoder = Decoder(3, n_subs, n_feats)
        n_colors = frame * 3  # Number of spectral channels (RGB channels multiplied by frames)
        
        self.flow_net = Flow_BasicBlock(in_channels=6, out_channels=3)
        
        # Calculate the number of branch networks
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        self.start_idx = []
        self.end_idx = []
        
        # Main trunk network and final convolution layer
        self.trunk = BranchUnit(n_colors, n_feats=16, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1, 
                                up_scale=1, conv=default_conv, use_tail=False)
        self.final = nn.Conv2d(16, n_colors, kernel_size=3, padding=1)
        
        # Determine the start and end index for each branch
        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)
            
    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward
    
    def encode(self, x):
        b, t, c, h, w = x.shape
        device = x.device
        max_frames = self.frame
        
        z_list_total = []

        # Process in fixed chunks of max_frames
        for start in range(0, t, max_frames):
            end = start + max_frames
            x_chunk = x[:, start:end, :, :, :]

            # If the chunk exceeds the input time dimension, we need to pad the last chunk
            if end > t:
                # Use the last available frame for padding
                last_frame = x[:, t - 1, :, :, :].unsqueeze(1)  # Get the last frame and expand its dimensions
                padding = end - t
                x_chunk = torch.cat([x_chunk, last_frame.repeat(1, padding, 1, 1, 1)], dim=1)  # Padding with last frame
            # Call the original forward logic
            z_list_chunk = self.encode_chunk(x_chunk, device)
            z_list_total.append(z_list_chunk)
        return z_list_total


    def decode(self,x, x_size, z_list_total):
        b, t, c, h, w = x_size.shape
        device = x.device
        max_frames = self.frame
        y_out_total = []
        
        for z_list in z_list_total:
            y_out_chunk = self.decode_chunk(x_size,z_list)
            y_out_total.append(y_out_chunk)
        
        # Concatenate the outputs from all chunks
        y_out = torch.cat(y_out_total, dim=1)  # Concatenate along the time dimension       
        # Remove duplicate frames at the end if necessary
        if t % max_frames != 0:
            overlap_frames = max_frames - (t % max_frames)
            # print(overlap_frames, y_out.shape)
            y_out = y_out[:, :-overlap_frames, :, :, :]  # Remove the last 'overlap_frames' time steps
            
        # ---- Apply color correction to y_out ----
        n, t, c, h, w = y_out.shape
        corrected_y_out = []  # 用于存储校正后的时间帧

        for i in range(t):  # 遍历时间维度
            # 对第 i 帧进行颜色校正
            corrected_frame = color_correction(
                lr_input=x[:, i, :, :, :],  # 第 i 帧的 LR 输入
                sr_output=y_out[:, i, :, :, :]  # 第 i 帧的 SR 输出
            )
            corrected_y_out.append(corrected_frame.unsqueeze(1))  # 恢复时间维度后存储

        # 将校正后的帧沿时间维度拼接
        corrected_y_out = torch.cat(corrected_y_out, dim=1)  # (n, t, c, h, w)
            
        return corrected_y_out


    def decode_chunk(self, x, z_list):
        b, t, c, h, w = x.shape
        device = x.device
        # x = x.view(b, self.frame * c, h, w)
        channel_counter = torch.zeros(self.frame * c).to(device)
        y = torch.zeros(b, self.frame * c, h, w).to(device)

        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]
            output_i = self.Decoder(z_list[g])
            y[:, sta_ind:end_ind, :, :] += output_i
            channel_counter[sta_ind:end_ind] += 1

        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        y1 = self.trunk(y)
        y1 = self.final(y1)
        y_out = y1 + y
        y_out = y_out.view(b, self.frame, c, h, w)
        return y_out


    def forward(self, x):
        b, t, c, h, w = x.shape
        device = x.device
        max_frames = self.frame
        
        y_out_total = []
        z_list_total = []

        # Process in fixed chunks of max_frames
        for start in tqdm(range(0, t, max_frames)):
            end = start + max_frames
            x_chunk = x[:, start:end, :, :, :]

            # If the chunk exceeds the input time dimension, we need to pad the last chunk
            if end > t:
                # Use the last available frame for padding
                last_frame = x[:, t - 1, :, :, :].unsqueeze(1)  # Get the last frame and expand its dimensions
                padding = end - t
                x_chunk = torch.cat([x_chunk, last_frame.repeat(1, padding, 1, 1, 1)], dim=1)  # Padding with last frame

            # Call the original forward logic
            y_out_chunk, z_list_chunk = self._process_chunk(x_chunk, device)

            y_out_total.append(y_out_chunk)
            z_list_total.append(z_list_chunk)

        # Concatenate the outputs from all chunks
        y_out = torch.cat(y_out_total, dim=1)  # Concatenate along the time dimension
        z_list = [z for sublist in z_list_total for z in sublist]  # Flatten the list of latent variables
        
        # 保存到指定路径
        # save_tensors_as_images(z_list, output_dir="/data1/userhome/luwen/Code/wzy/CAD2VSR/demo_GAE_fig", prefix="tensor_image")

        # Remove duplicate frames at the end if necessary
        if t % max_frames != 0:
            overlap_frames = max_frames - (t % max_frames)
            # print(overlap_frames, y_out.shape)
            y_out = y_out[:, :-overlap_frames, :, :, :]  # Remove the last 'overlap_frames' time steps

        # ---- Apply color correction to y_out ----
        n, t, c, h, w = y_out.shape
        corrected_y_out = []  # 用于存储校正后的时间帧

        for i in range(t):  # 遍历时间维度
            # 对第 i 帧进行颜色校正
            corrected_frame = color_correction(
                lr_input=x[:, i, :, :, :],  # 第 i 帧的 LR 输入
                sr_output=y_out[:, i, :, :, :]  # 第 i 帧的 SR 输出
            )
            corrected_y_out.append(corrected_frame.unsqueeze(1))  # 恢复时间维度后存储

        # 将校正后的帧沿时间维度拼接
        corrected_y_out = torch.cat(corrected_y_out, dim=1)  # (n, t, c, h, w)
        # print(corrected_y_out.shape, y_out.shape)

        # return y_out, z_list
        return corrected_y_out
        # return y_out


    def encode_chunk(self, x, device):
        b, t, c, h, w = x.shape
        self.check_if_mirror_extended(x)
        flows_forward, flows_backward = self.compute_flow(x)

        outputs = []
        feat_prop = x.new_zeros(b, 3, h, w).to(device)
        for i in range(t):
            lr_curr = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :] if flows_forward is not None else flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            feat_prop = self.flow_net(feat_prop)
            outputs.append(feat_prop)

        outputs = torch.stack(outputs, dim=1)
        x = outputs.view(b, self.frame * c, h, w)
        z_list = []
        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]
            xi = x[:, sta_ind:end_ind, :, :]
            z = self.Encoder(xi)
            z_list.append(z)
        return z_list



    def _process_chunk(self, x, device):
        b, t, c, h, w = x.shape
        self.check_if_mirror_extended(x)
        flows_forward, flows_backward = self.compute_flow(x)

        outputs = []
        feat_prop = x.new_zeros(b, 3, h, w).to(device)
        for i in range(t):
            lr_curr = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :] if flows_forward is not None else flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            feat_prop = self.flow_net(feat_prop)
            outputs.append(feat_prop)

        outputs = torch.stack(outputs, dim=1)
        x = outputs.view(b, t * c, h, w)
        
        # x = x.view(b, t * c, h, w)

        channel_counter = torch.zeros(t * c).to(device)
        y = torch.zeros(b, t * c, h, w).to(device)
        z_list = []

        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]
            xi = x[:, sta_ind:end_ind, :, :]
            z = self.Encoder(xi)
            z_list.append(z)
            output_i = self.Decoder(z)
            y[:, sta_ind:end_ind, :, :] += output_i
            channel_counter[sta_ind:end_ind] += 1

        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        y1 = self.trunk(y)
        y1 = self.final(y1)
        y_out = y1 + y
        y_out = y_out.view(b, t, c, h, w)

        return y_out, z_list




class SPyNet(BaseModule):
    """SPyNet network structure.

    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(BaseModule):
    """Basic Module for SPyNet.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)


# network = GAE()
# tensor_size = (32, 15, 3, 128, 128) 
# input_tensor = torch.randn(tensor_size)

# # 将 tensor 输入网络
# outputs = network(input_tensor)
# z_list = network.encode(input_tensor)
# outputs_decode = network.decode(input_tensor, z_list)

# # 打印输出的尺寸
# print(len(z_list))
# for i in range (len(z_list)):
#     print("Output shape:", outputs.shape, outputs_decode.shape)
#     print("feats_sq shape:", z_list[i].shape)


import torch
import os
from torchvision.utils import save_image

def save_tensors_as_images(z_list, output_dir, prefix="image"):
    """
    保存 PyTorch 张量列表为图片。
    
    Args:
        z_list (list): 包含 PyTorch 张量的列表。
        output_dir (str): 保存图片的目录路径。
        prefix (str): 图片文件名前缀。
    """
    # 创建保存目录
    os.makedirs(output_dir, exist_ok=True)

    for idx, tensor in enumerate(z_list):
        # 检查是否为 PyTorch 张量
        if not isinstance(tensor, torch.Tensor):
            print(f"第 {idx} 个元素不是 PyTorch 张量，跳过...")
            continue
        
        # 确保张量为 3 通道 (如果是单通道图像，添加通道维度)
        if tensor.dim() == 2:  # [H, W] -> [1, H, W]
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 4:  # 如果是 [B, C, H, W] 只取第一张
            tensor = tensor[0]
        
        # 归一化到 [0, 1]，防止保存图像变黑
        # tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-5)

        # 保存图像
        save_path = os.path.join(output_dir, f"{prefix}_{idx:05d}.png")
        save_image(tensor, save_path)
        print(f"保存图片到: {save_path}")
        
        
def color_correction(lr_input, sr_output):
    """
    Align the color statistics (mean and variance) of the SR image with the LR input.

    Args:
    - lr_input (Tensor): (C, H, W) or (N, C, H, W), the low-resolution input image.
    - sr_output (Tensor): (C, H*scale, W*scale) or (N, C, H*scale, W*scale), the super-resolution output.

    Returns:
    - Tensor: Same shape as sr_output, the color-corrected output image.
    """
    # Check input dimensions and ensure compatibility
    if lr_input.dim() != sr_output.dim():
        raise ValueError("The dimensions of lr_input and sr_output must match.")

    # Calculate spatial dimensions for mean/std computation
    dims = tuple(range(2, lr_input.dim()))  # Compute per-channel statistics

    # Calculate mean and standard deviation
    sr_mean = sr_output.mean(dim=dims, keepdim=True)
    sr_std = sr_output.std(dim=dims, keepdim=True)
    lr_mean = lr_input.mean(dim=dims, keepdim=True)
    lr_std = lr_input.std(dim=dims, keepdim=True)

    # Prevent division by zero by clamping standard deviations
    sr_std = sr_std.clamp(min=1e-6)
    lr_std = lr_std.clamp(min=1e-6)

    # Perform color correction
    corrected_output = (sr_output - sr_mean) / sr_std * lr_std + lr_mean

    # Clamp values to valid range [0, 1]
    corrected_output = corrected_output.clamp(0.0, 1.0)

    return corrected_output

