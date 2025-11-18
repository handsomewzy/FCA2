import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    # print(pred.shape, target.shape)
    return torch.sqrt((pred - target)**2 + eps)

@weighted_loss
def charbonnier_loss_gae(pred, target, eps=1e-12):
    # print(pred.shape, target.shape)
    return torch.sqrt((pred - target)**2 + eps)

@LOSS_REGISTRY.register()
class ComisrLoss(nn.Module):
    def __init__(self, beta=20.0, gamma=1.0):
        super(ComisrLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, gt):
        """
        计算损失

        Args:
            inputs: 包含 (gen_output, pre_gen, pre_warp_hi, inputs_raw) 的元组
            IHR: 高分辨率真实图像，形状为 (B, C, H, W)
            ILR: 低分辨率真实图像，形状为 (B, C, H, W)
            N: 总帧数

        Returns:
            总损失 Ltotal
        """
        # 从元组中提取各个元素
        outputs_fwd, outputs_bwd, pre_warp_hi_fwd_list, pre_warp_hi_bwd_list, inputs_raw= inputs
        IHR = gt
        ILR = inputs_raw
        L2_loss = MSELoss()

        # 计算高分辨率内容损失
        LHR_content = L2_loss(IHR, outputs_fwd) + L2_loss(IHR, outputs_bwd)

        # 计算低分辨率变形损失
        LLR_warp = L2_loss(IHR, pre_warp_hi_fwd_list) + L2_loss(IHR, pre_warp_hi_bwd_list)

        # 计算总损失
        Ltotal = self.beta * LHR_content + self.gamma * LLR_warp

        return Ltotal
    
@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)



@LOSS_REGISTRY.register()
class CharbonnierLoss_gae(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss_gae, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_list, target_list, weight=None, **kwargs):
        """
        Args:
            pred_list (list of Tensors): 每个 Tensor 形状为 (N, C, H, W)，表示预测值列表。
            target_list (list of Tensors): 每个 Tensor 形状为 (N, C, H, W)，表示目标值列表。
            weight (Tensor, optional): 形状为 (N, C, H, W) 的元素权重。默认：None。
        """
        total_loss = 0.0
        num_elements = len(pred_list)

        for pred, target in zip(pred_list, target_list):
            # 逐元素计算 Charbonnier loss 并累加
            total_loss += charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)

        # 取平均（或直接返回累加的总损失值）
        return self.loss_weight * (total_loss / num_elements)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss



@LOSS_REGISTRY.register()
class bi_CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=[0.5, 0.25, 0.25], reduction='mean', eps=1e-12):
        super(bi_CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        loss = self.loss_weight[0] * charbonnier_loss(pred[0], target, weight, eps=self.eps, reduction=self.reduction) + \
               self.loss_weight[1] * charbonnier_loss(pred[1], target, weight, eps=self.eps, reduction=self.reduction) + \
               self.loss_weight[2] * charbonnier_loss(pred[2], target, weight, eps=self.eps, reduction=self.reduction)
        return loss

@LOSS_REGISTRY.register()
class bi_unsym_CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=[0.5, 0.25, 0.25], reduction='mean', eps=1e-12):
        super(bi_unsym_CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        lr_bic = pred[3]
        loss_hr = self.loss_weight[0] * charbonnier_loss(pred[0], target, weight, eps=self.eps, reduction=self.reduction)
        loss_b = self.loss_weight[2] * charbonnier_loss(pred[2], 0.5*target + 0.5*lr_bic, weight, eps=self.eps, reduction=self.reduction)
        loss_f = self.loss_weight[1] * charbonnier_loss(pred[1], target, weight, eps=self.eps, reduction=self.reduction)
        loss = loss_hr + loss_b + loss_f
        return loss


@LOSS_REGISTRY.register()
class bi_SD_CharbonnierLoss(nn.Module):
    def __init__(self, loss_weight=[0.5, 0.25, 0.25], reduction='mean', eps=1e-12):
        super(bi_SD_CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        b, n, _, h, w = target.size()
        # S
        Ss = []
        for i in range(n):
            frm = target[:,i,:,:,:]
            S = F.interpolate(frm, scale_factor=0.5, mode='bilinear', align_corners=False)
            S = F.interpolate(S, scale_factor=2, mode='bilinear', align_corners=False)
            Ss.append(S)
        S = torch.stack(Ss,dim=1)

        loss_hr = self.loss_weight[0] * charbonnier_loss(pred[0], target, weight, eps=self.eps, reduction=self.reduction)
        loss_b = self.loss_weight[2] * charbonnier_loss(pred[2], S, weight, eps=self.eps, reduction=self.reduction)
        loss_f = self.loss_weight[1] * charbonnier_loss(pred[1], target, weight, eps=self.eps, reduction=self.reduction)
        loss = loss_hr + loss_b + loss_f
        return loss


@LOSS_REGISTRY.register()
class bi_SDs_CharbonnierLoss(nn.Module):
    def __init__(self, loss_weight=[0.5, 0.25, 0.25], reduction='mean', eps=1e-12):
        super(bi_SDs_CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        b, n, _, h, w = target.size()
        # S, lr
        Ss = []
        Bis = []
        for i in range(n):
            frm = target[:,i,:,:,:]
            S = F.interpolate(frm, scale_factor=0.5, mode='bilinear', align_corners=False)
            S = F.interpolate(S, scale_factor=2, mode='bilinear', align_corners=False)
            Ss.append(S)
            Bis.append(F.interpolate(S, scale_factor=0.25, mode='bilinear', align_corners=False))
        S = torch.stack(Ss,dim=1)
        bi = torch.stack(Bis,dim=1)

        hr, f, b, s = pred

        loss_hr = self.loss_weight[0] * charbonnier_loss(hr, target, weight, eps=self.eps, reduction=self.reduction)
        loss_b = self.loss_weight[2] * charbonnier_loss(b, S, weight, eps=self.eps, reduction=self.reduction)
        loss_f = self.loss_weight[1] * charbonnier_loss(f, target, weight, eps=self.eps, reduction=self.reduction)
        loss_s = self.loss_weight[3] * charbonnier_loss(s, bi, weight, eps=self.eps, reduction=self.reduction)
        loss = loss_hr + loss_b + loss_f + loss_s
        return loss
