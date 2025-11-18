# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.model import BaseModule

# from mmagic.registry import MODELS
# from ..basicvsr.basicvsr_net import BasicVSRNet, ResidualBlocksWithInputConv

from .basicvsr_arch import ResidualBlocksWithInputConv, SPyNet, BasicVSRNet_CAV

from basicsr.utils.registry import ARCH_REGISTRY
from .proposedGAE_arch import GAE
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer, ResidualBlockNoBN_sft_1x1, SFTLayer_torch_1x1, SFTLayer_torch_3x3
from .component.degradation_aware import Ranker_128_up_WOIBP


@ARCH_REGISTRY.register()
class CrfClean(BaseModule):
    """RealBasicVSR network structure for real-world video super-resolution.

    Support only x4 upsampling.

    Paper:
        Investigating Tradeoffs in Real-World Video Super-Resolution, arXiv

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_propagation_blocks (int, optional): Number of residual blocks in
            each propagation branch. Default: 20.
        num_cleaning_blocks (int, optional): Number of residual blocks in the
            image cleaning module. Default: 20.
        dynamic_refine_thres (int, optional): Stop cleaning the images when
            the residue is smaller than this value. Default: 255.
        is_fix_cleaning (bool, optional): Whether to fix the weights of
            the image cleaning module during training. Default: False.
        is_sequential_cleaning (bool, optional): Whether to clean the images
            sequentially. This is used to save GPU memory, but the speed is
            slightly slower. Default: False.
    """

    def __init__(self,
                 mid_channels=64,
                 num_cleaning_blocks=20,
                 dynamic_refine_thres=255,
                 is_fix_cleaning=False,
                 is_sequential_cleaning=False):

        super().__init__()

        self.dynamic_refine_thres = dynamic_refine_thres / 255.
        self.is_sequential_cleaning = is_sequential_cleaning

        # image cleaning module
        self.image_cleaning = nn.Sequential(
            ResidualBlocksWithInputConv(3, mid_channels, num_cleaning_blocks),
            nn.Conv2d(mid_channels, 3, 3, 1, 1, bias=True),
        )

        if is_fix_cleaning:  # keep the weights of the cleaning module fixed
            self.image_cleaning.requires_grad_(False)

    def forward(self, lqs, return_lqs=False):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            return_lqs (bool): Whether to return LQ sequence. Default: False.

        Returns:
            Tensor: Output HR sequence.
        """
        n, t, c, h, w = lqs.size()

        for _ in range(0, 3):  # at most 3 cleaning, determined empirically
            if self.is_sequential_cleaning:
                residues = []
                for i in range(0, t):
                    residue_i = self.image_cleaning(lqs[:, i, :, :, :])
                    lqs[:, i, :, :, :] += residue_i
                    residues.append(residue_i)
                residues = torch.stack(residues, dim=1)
            else:  # time -> batch, then apply cleaning at once
                lqs = lqs.view(-1, c, h, w)
                residues = self.image_cleaning(lqs)
                lqs = (lqs + residues).view(n, t, c, h, w)

            # determine whether to continue cleaning
            if torch.mean(torch.abs(residues)) < self.dynamic_refine_thres:
                break
        outputs = lqs

        if return_lqs:
            return outputs, lqs
        else:
            return outputs
        

@ARCH_REGISTRY.register()
class CAD2VSR(BaseModule):
    """RealBasicVSR network structure for real-world video super-resolution.

    Support only x4 upsampling.

    Paper:
        Investigating Tradeoffs in Real-World Video Super-Resolution, arXiv

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_propagation_blocks (int, optional): Number of residual blocks in
            each propagation branch. Default: 20.
        num_cleaning_blocks (int, optional): Number of residual blocks in the
            image cleaning module. Default: 20.
        dynamic_refine_thres (int, optional): Stop cleaning the images when
            the residue is smaller than this value. Default: 255.
        spynet_pretrained (str, optional): Pre-trained model path of SPyNet.
            Default: None.
        is_fix_cleaning (bool, optional): Whether to fix the weights of
            the image cleaning module during training. Default: False.
        is_sequential_cleaning (bool, optional): Whether to clean the images
            sequentially. This is used to save GPU memory, but the speed is
            slightly slower. Default: False.
    """

    def __init__(self,
                 mid_channels=64,
                 num_propagation_blocks=20,
                 num_cleaning_blocks=20,
                 dynamic_refine_thres=255,
                 spynet_pretrained=None,
                 is_fix_cleaning=False,
                 is_sequential_cleaning=False):

        super().__init__()
        self.dynamic_refine_thres = dynamic_refine_thres / 255.
        self.is_sequential_cleaning = is_sequential_cleaning
        
        self.model_GAE = GAE(spynet_pretrained='/data1/userhome/luwen/Code/wzy/CAD2VSR/spynet_20210409-c6c1bd09.pth', n_subs=9, n_ovls=3)
        # 先加载 .pth 文件内容
        load_net = torch.load('/data1/userhome/luwen/Code/wzy/CAD2VSR/experiments/vid4_gae93_color/models/net_g_70000.pth')
        load_net = load_net["params"]
        # 然后将加载的字典传递给 model_GAE
        self.model_GAE.load_state_dict(load_net)
        
        # 加载CAVSR中的encoder，后续去除其中的IBP等code元素，再经过编码传入basicVSR
        self.encoder = Ranker_128_up_WOIBP()
        checkpoint = torch.load(r"./ranker.pth")
        self.encoder.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
        print('encoder model loading done!')  
        
        num_DA_block = 5
        self.modulate_b = sft_net(num_in_ch=3, num_out_ch=3, rep_feat=128, num_block=num_DA_block)
        self.modulate_f = sft_net(num_in_ch=3, num_out_ch=3, rep_feat=128, num_block=num_DA_block)   

        # image cleaning module
        self.image_cleaning = nn.Sequential(
            ResidualBlocksWithInputConv(3, mid_channels, num_cleaning_blocks),
            nn.Conv2d(mid_channels, 3, 3, 1, 1, bias=True),
        )

        if is_fix_cleaning:  # keep the weights of the cleaning module fixed
            self.image_cleaning.requires_grad_(False)

        # BasicVSR
        self.basicvsr = BasicVSRNet_CAV(mid_channels, num_propagation_blocks,
                                    spynet_pretrained)
        self.basicvsr.spynet.requires_grad_(False)

    def forward(self, lqs, return_lqs=False):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            return_lqs (bool): Whether to return LQ sequence. Default: False.

        Returns:
            Tensor: Output HR sequence.
        """
        n, t, c, h, w = lqs.size()
        lqs_ori = lqs
        device = lqs.device
        gt_size = torch.zeros((n, t, c, 4 * h, 4 * w)).to(device)
        outputs = []

        for _ in range(0, 3):  # at most 3 cleaning, determined empirically
            if self.is_sequential_cleaning:
                residues = []
                for i in range(0, t):
                    residue_i = self.image_cleaning(lqs[:, i, :, :, :])
                    lqs[:, i, :, :, :] += residue_i
                    residues.append(residue_i)
                residues = torch.stack(residues, dim=1)
            else:  # time -> batch, then apply cleaning at once
                lqs = lqs.view(-1, c, h, w)
                residues = self.image_cleaning(lqs)
                lqs = (lqs + residues).view(n, t, c, h, w)

            # determine whether to continue cleaning
            if torch.mean(torch.abs(residues)) < self.dynamic_refine_thres:
                break


        # training status of encoder must be set as eval!!!
        self.encoder.eval()
        if self.encoder.training == 'train':
            raise ValueError('training status of encoder must be set as eval')
        rep_list = []
        with torch.no_grad():
            for i in range(t):
                rep_list.append(self.encoder(lqs[:,i]))
        rep = torch.stack(rep_list, dim=1)
        # print(rep.shape) # torch.Size([8, 7, 128, 64, 64])


        x_b_feat = []
        for i in range(t):
            fea = self.modulate_b(lqs_ori[:,i], rep[:,i])
            x_b_feat.append(fea)
        x_b_feat = torch.stack(x_b_feat, dim=1)

        x_f_feat = []
        for i in range(t):
            fea = self.modulate_f(lqs_ori[:,i], rep[:,i])
            x_f_feat.append(fea)
        x_f_feat = torch.stack(x_f_feat, dim=1)
        # print(x_b_feat.shape, x_f_feat.shape) # torch.Size([8, 7, 3, 64, 64]) torch.Size([8, 7, 3, 64, 64])


        # Super-resolution (BasicVSR)
        self.model_GAE.eval()
        lqs_sub = self.model_GAE.encode(lqs)
        lqs_sublist = [torch.stack(tensor_list, dim=1) for tensor_list in lqs_sub]
        
        x_b_feat_sub =self.model_GAE.encode(x_b_feat)
        x_b_feat_sublist = [torch.stack(tensor_list, dim=1) for tensor_list in x_b_feat_sub]
        x_f_feat_sub =self.model_GAE.encode(x_f_feat)
        x_f_feat_sublist = [torch.stack(tensor_list, dim=1) for tensor_list in x_f_feat_sub]
        
        for i in range(len(lqs_sublist)):
            # print(lqs.shape) # torch.Size([8, 3, 3, 64, 64])
            _outputs = self.basicvsr(lqs_sublist[i], x_f_feat_sublist[i], x_b_feat_sublist[i])
            outputs.append(_outputs)

        # decode to get final video
        original_lqs_sub = [list(torch.unbind(tensor, dim=1)) for tensor in outputs]
        f_outputs = self.model_GAE.decode(lqs, gt_size, original_lqs_sub)

        if return_lqs:
            return outputs, lqs
        else:
            return f_outputs
        
        
        
class sft_net(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, rep_feat=256, num_block=15):
        super().__init__()
        self.main = nn.Sequential(make_layer(ResidualBlockNoBN_sft_1x1, num_block, num_feat=num_out_ch, rep_feat=rep_feat))
        #self.sft = SFTLayer_torch_1x1(rep_feat, num_out_ch)
        self.conv2 = nn.Conv2d(num_out_ch, num_out_ch, 3, 1, 1, bias=True)
    def forward(self, fea, rep):
        x = fea.clone()
        res = self.main((fea, rep))
        #res = self.sft((res))
        res = self.conv2(res[0])
        return fea + x