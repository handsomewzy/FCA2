import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Define LeakyReLU
def lrelu(inputs, alpha):
    return F.leaky_relu(inputs, negative_slope=alpha)


# Preprocessing: [0, 1] => [-1, 1]
def preprocess(image):
    return image * 2 - 1


# Deprocessing: [-1, 1] => [0, 1]
def deprocess(image):
    return (image + 1) / 2


# Max Pooling
def maxpool(inputs, kernel_size=2):
    return F.max_pool2d(inputs, kernel_size=kernel_size)


# Conv2D Transpose (transposed convolution)
def conv2_tran(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True):
    return nn.ConvTranspose2d(
        in_channels=batch_input.size(1),
        out_channels=output_channel,
        kernel_size=kernel,
        stride=stride,
        padding=1,
        output_padding=1 if stride > 1 else 0,
        bias=use_bias
    )(batch_input)


# Conv2D
def conv2(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True):
    return nn.Conv2d(
        in_channels=batch_input.size(1),
        out_channels=output_channel,
        kernel_size=kernel,
        stride=stride,
        padding=1,
        bias=use_bias
    )(batch_input)


# Upscale bilinear interpolation
def upscale_x(inputs, scale=4):
    return F.interpolate(inputs, scale_factor=scale, mode='bilinear', align_corners=False)


# Bicubic interpolation
def bicubic_x(inputs, scale=4):
    return F.interpolate(inputs, scale_factor=scale, mode='bicubic', align_corners=False)


# Bicubic upscaling specifically for scale=4
def bicubic_four(inputs):
    """Bicubic four upscaling."""
    b, c, h, w = inputs.size()

    # Padding (top, bottom, left, right)
    p_inputs = F.pad(inputs, (1, 1, 1, 1), mode='replicate')  # Replicate padding

    # Reshape and apply the weights as in the original code
    r = 0.75
    mat = np.float32([[0, 1, 0, 0], [-r, 0, r, 0], [2 * r, r - 3, 3 - 2 * r, -r], [-r, 2 - r, r - 2, r]])
    weights = [np.float32([1.0, t, t ** 2, t ** 3]).dot(mat) for t in [0.0, 0.25, 0.5, 0.75]]

    # Apply the weights row-wise
    hi_res_array = []
    for hi in range(4):
        cur_wei = torch.tensor(weights[hi], dtype=torch.float32, device=inputs.device)
        cur_data = sum(cur_wei[i] * p_inputs[:, :, i:h+i, :] for i in range(4))
        hi_res_array.append(cur_data)

    hi_res_y = torch.stack(hi_res_array, dim=2).reshape(b, c, h * 4, w + 3)

    # Apply the weights column-wise
    hi_res_array = []
    for hj in range(4):
        cur_wei = torch.tensor(weights[hj], dtype=torch.float32, device=inputs.device)
        cur_data = sum(cur_wei[i] * hi_res_y[:, :, :, i:w+i] for i in range(4))
        hi_res_array.append(cur_data)

    hi_res = torch.stack(hi_res_array, dim=3).reshape(b, c, h * 4, w * 4)

    return hi_res
