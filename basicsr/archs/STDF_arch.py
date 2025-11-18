import torch
import torch.nn as nn
import torch.nn.functional as F
# from ops.dcn.deform_conv import ModulatedDeformConv
from basicsr.ops.stdf_ops.dcn.deform_conv import ModulatedDeformConv

# ==========
# Spatio-temporal deformable fusion module
# ==========

class STDF(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STDF, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)  # 不使用原地操作
        )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
                    nn.ReLU(inplace=True),  # 不使用原地操作
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True)  # 不使用原地操作
                )
            )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2*nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True),  # 不使用原地操作
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)  # 不使用原地操作
                )
            )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
            nn.ReLU(inplace=True),  # 不使用原地操作
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True),  # 不使用原地操作
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)  # 不使用原地操作
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)  # 不使用原地操作
        )

        # regression head
        self.offset_mask = nn.Conv2d(
            nf, in_nc*3*self.size_dk, base_ks, padding=base_ks//2
        )

        # deformable conv
        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks//2, deformable_groups=in_nc
        )

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # feature extraction (with downsampling)
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            
            # print(f"out shape: {out.shape}, out_lst[{i}] shape: {out_lst[i].shape}")
            # 假设需要对齐的维度是 H 和 W
            h_min = min(out.shape[2], out_lst[i].shape[2])
            w_min = min(out.shape[3], out_lst[i].shape[3])

            out = out[:, :, :h_min, :w_min]
            out_lst[i] = out_lst[i][:, :, :h_min, :w_min]

            # 然后执行 torch.cat
            out = up_conv(torch.cat([out, out_lst[i]], dim=1))
            
            # out = up_conv(
            #     torch.cat([out, out_lst[i]], 1)
            # )
            

        # compute offset and mask
        off_msk = self.offset_mask(self.out_conv(out))
        off = off_msk[:, :in_nc*2*n_off_msk, ...]
        msk = torch.sigmoid(
            off_msk[:, in_nc*2*n_off_msk:, ...]
        )

        # perform deformable convolutional fusion
        fused_feat = F.relu(
            self.deform_conv(inputs.contiguous(), off.contiguous(), msk.contiguous()), 
            inplace=True  # 使用非原地操作
        )

        return fused_feat



