import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops import deprocess, upscale_x
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class COMISR(nn.Module):
    def __init__(self, vsr_scale=4, num_resblock=10):
        super(COMISR, self).__init__()
        # Initialize hidden states for forward and backward directions (per batch)
        self.pre_inputs_fwd = None
        self.pre_gen_fwd = None
        self.pre_warp_fwd = None

        self.pre_inputs_bwd = None
        self.pre_gen_bwd = None
        self.pre_warp_bwd = None
        self.vsr_scale = vsr_scale
        self.num_resblock = num_resblock

        # Define generator and fnet models
        self.generator = GeneratorF(3, num_resblock, vsr_scale)
        self.fnet = FNet()

        # Convolution transpose for deconv_flow layers
        self.deconv_flow_tran1 = nn.ConvTranspose2d(2, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv_flow_tran2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv_flow_conv = nn.Conv2d(64, 2, kernel_size=3, padding=1)

    # def forward_frame(self, inputs_raw, pre_inputs, pre_gen, pre_warp):
    #     """
    #     Process a single frame through the VSR pipeline
    #     """
    #     device = inputs_raw.device
        
    #     # Space-to-Depth operation on pre_warp (Pixel unshuffle)
    #     transpose_pre = F.pixel_unshuffle(pre_warp.to(device), downscale_factor=self.vsr_scale)

    #     # Concatenate inputs
    #     inputs_all = torch.cat([inputs_raw, transpose_pre], dim=1)

    #     # Generator pass
    #     gen_output = self.generator(inputs_all)

    #     # Deprocess the generated output and assign it to pre_gen
    #     pre_gen = self.deprocess(gen_output)

    #     # Frame-to-frame operations for fnet
    #     inputs_frames = torch.cat([pre_inputs, inputs_raw], dim=1)
    #     gen_flow_lr = self.fnet(inputs_frames)

    #     # Padding the generated flow
    #     gen_flow_lr_padded = F.pad(gen_flow_lr, (1, 1, 1, 1), mode='reflect')

    #     # Deconvolution (Upscaling)
    #     deconv_flow = F.relu(self.deconv_flow_tran1(gen_flow_lr_padded))
    #     deconv_flow = F.relu(self.deconv_flow_tran2(deconv_flow))
    #     deconv_flow = self.deconv_flow_conv(deconv_flow)

    #     # Upscale gen_flow_lr and combine with deconv_flow
    #     gen_flow = upscale_x(gen_flow_lr * 4.0, scale=self.vsr_scale)
    #     gen_flow = deconv_flow + gen_flow

    #     # Apply dense_image_warp equivalent using grid_sample for pre_warp_hi
    #     pre_warp_hi = self.dense_image_warp(pre_gen, gen_flow)
    #     pre_warp_hi = pre_warp_hi + self.extract_detail(pre_warp_hi)

    #     return gen_output, pre_gen, pre_warp_hi, inputs_raw

    def dense_image_warp(self, img, flow):
        # Dense image warp using bilinear sampling (similar to TensorFlow's dense_image_warp)
        b, c, h, w = img.size()
        grid = self.flow_to_grid(flow, h, w)
        return F.grid_sample(img, grid, mode='bilinear', align_corners=False)

    def flow_to_grid(self, flow, height, width):
        # Convert optical flow to grid
        b, _, h, w = flow.size()
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, w), torch.linspace(-1, 1, h))
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1).to(flow.device)
        flow = flow.permute(0, 2, 3, 1)
        return grid + flow  # Adjust grid based on flow

    def extract_detail(self, img):
        # Custom detail extraction (dummy example, replace with real operation)
        return img * 0.1  # Adjust this logic based on your extraction ops
    
    def forward_frame(self, inputs_raw, pre_inputs, pre_gen, pre_warp):
        """
        Process a single frame through the VSR pipeline
        """
        device = inputs_raw.device
        
        # Space-to-Depth operation on pre_warp (Pixel unshuffle)
        transpose_pre = F.pixel_unshuffle(pre_warp.to(device), downscale_factor=self.vsr_scale)

        # Concatenate inputs
        inputs_all = torch.cat([inputs_raw, transpose_pre], dim=1)

        # Generator pass
        gen_output = self.generator(inputs_all)

        # Deprocess the generated output and assign it to pre_gen
        pre_gen = deprocess(gen_output)

        # Frame-to-frame operations for fnet
        inputs_frames = torch.cat([pre_inputs, inputs_raw], dim=1)
        gen_flow_lr = self.fnet(inputs_frames)

        # Padding the generated flow
        # gen_flow_lr_padded = F.pad(gen_flow_lr, (1, 1, 1, 1), mode='reflect')

        # Deconvolution (Upscaling)
        deconv_flow = F.relu(self.deconv_flow_tran1(gen_flow_lr))
        deconv_flow = F.relu(self.deconv_flow_tran2(deconv_flow))
        deconv_flow = self.deconv_flow_conv(deconv_flow)

        # Upscale gen_flow_lr and combine with deconv_flow
        gen_flow = upscale_x(gen_flow_lr * 4.0, scale=self.vsr_scale)
        # # 在执行操作之前检查形状
        # print(f"pre_gen 的形状: {pre_gen.shape}")
        # print(f"gen_flow 的形状: {gen_flow.shape}")
        # print(f"deconv_flow 的形状: {deconv_flow.shape}")

        gen_flow = deconv_flow + gen_flow

        # Apply dense_image_warp equivalent using grid_sample for pre_warp_hi
        pre_warp_hi = self.dense_image_warp(pre_gen, gen_flow)
        pre_warp_hi = pre_warp_hi + self.extract_detail(pre_warp_hi)

        return gen_output, pre_gen, pre_warp_hi, inputs_raw

    def forward(self, inputs_raw_seq, mode='train'):
        """
        Process a sequence of frames (batch_size, t, c, h, w) forward and backward.
        """
        # print(inputs_raw_seq.shape)
        batch_size, t, c, h, w = inputs_raw_seq.shape
        
        # Initialize outputs for forward and backward passes
        outputs_fwd = []
        outputs_bwd = []
        pre_warp_hi_fwd_list = []
        pre_warp_hi_bwd_list = []

        # Initialize hidden states for each batch
        if self.pre_inputs_fwd is None:
            self.pre_inputs_fwd = torch.zeros((batch_size, c, h, w), device=inputs_raw_seq.device)
            self.pre_gen_fwd = torch.zeros((batch_size, c, 4*h, 4*w), device=inputs_raw_seq.device)
            self.pre_warp_fwd = torch.zeros((batch_size, c, 4*h, 4*w), device=inputs_raw_seq.device)

        if self.pre_inputs_bwd is None:
            self.pre_inputs_bwd = torch.zeros((batch_size, c, h, w), device=inputs_raw_seq.device)
            self.pre_gen_bwd = torch.zeros((batch_size, c, 4*h, 4*w), device=inputs_raw_seq.device)
            self.pre_warp_bwd = torch.zeros((batch_size, c, 4*h, 4*w), device=inputs_raw_seq.device)

        ### Forward Pass ###
        for i in range(t):
            frame = inputs_raw_seq[:, i]  # Get frame i for all batches
            # Process current frame for each batch
            output_fwd, self.pre_gen_fwd, self.pre_warp_fwd, self.pre_inputs_fwd = self.forward_frame(
                frame, self.pre_inputs_fwd, self.pre_gen_fwd, self.pre_warp_fwd)
            outputs_fwd.append(output_fwd)
            pre_warp_hi_fwd_list.append(self.pre_warp_fwd)

        ### Backward Pass (optional) ###
        for i in reversed(range(t)):
            frame = inputs_raw_seq[:, i]  # Get frame i for all batches
            # Process current frame in reverse order for each batch
            output_bwd, self.pre_gen_bwd, self.pre_warp_bwd, self.pre_inputs_bwd = self.forward_frame(
                frame, self.pre_inputs_bwd, self.pre_gen_bwd, self.pre_warp_bwd)
            outputs_bwd.insert(0, output_bwd)  # Insert at the beginning for reverse order
            pre_warp_hi_bwd_list.insert(0, self.pre_warp_bwd)  # Insert at the beginning for reverse order

        # Convert list of tensors to a single tensor
        outputs_fwd = torch.stack(outputs_fwd, dim=1)  # (batch_size, t, c, h, w)
        outputs_bwd = torch.stack(outputs_bwd, dim=1)  # (batch_size, t, c, h, w)
        pre_warp_hi_fwd_list = torch.stack(pre_warp_hi_fwd_list, dim=1)  # (batch_size, t, c, h, w)
        pre_warp_hi_bwd_list = torch.stack(pre_warp_hi_bwd_list, dim=1)  # (batch_size, t, c, h, w)

        if mode == 'train':
            # return outputs_fwd, outputs_bwd, pre_warp_hi_fwd_list, pre_warp_hi_bwd_list, inputs_raw_seq
            return outputs_fwd  # 返回第一个前向输出
        elif mode == 'test':
            return outputs_fwd  # 返回第一个前向输出


class FNet(nn.Module):
    def __init__(self):
        super(FNet, self).__init__()
        # Define down blocks
        self.encoder_1 = self.down_block(6, 32)
        self.encoder_2 = self.down_block(32, 64)
        self.encoder_3 = self.down_block(64, 128)
        # Define up blocks
        self.decoder_1 = self.up_block(128, 256)
        self.decoder_2 = self.up_block(256, 128)
        self.decoder_3 = self.up_block(128, 64)
        # Output stage
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)

    def down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )
    
    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        # Downsample
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)
        # Upsample
        x = F.interpolate(self.decoder_1(x), scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(self.decoder_2(x), scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(self.decoder_3(x), scale_factor=2, mode='bilinear', align_corners=False)
        # Output stage
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = torch.tanh(self.conv2(x)) * 24.0  # 24.0 is the max velocity, from TecoGAN paper
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class GeneratorFEncoder(nn.Module):
    def __init__(self, num_resblock=10):
        super(GeneratorFEncoder, self).__init__()
        self.input_stage = nn.Conv2d(51, 64, kernel_size=3, padding=1)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_resblock)])

    def forward(self, x):
        x = F.relu(self.input_stage(x))
        return self.residual_blocks(x)


class GeneratorFDecoder(nn.Module):
    def __init__(self, gen_output_channels, vsr_scale):
        super(GeneratorFDecoder, self).__init__()
        self.vsr_scale = vsr_scale
        self.conv_tran1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_tran2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.output_stage = nn.Conv2d(64, gen_output_channels, kernel_size=3, padding=1)

    def forward(self, x, gen_inputs):
        # Upsample to high resolution
        if self.vsr_scale == 2:
            x = F.relu(self.conv_tran1(x))
        elif self.vsr_scale == 4:
            x = F.relu(self.conv_tran1(x))
            x = F.relu(self.conv_tran2(x))

        # Output stage
        low_res_in = gen_inputs[:, :3, :, :]  # Ignore warped pre high res
        bicubic_hi = F.interpolate(low_res_in, scale_factor=self.vsr_scale, mode='bicubic', align_corners=False)
        x = self.output_stage(x) + bicubic_hi
        return x


class GeneratorF(nn.Module):
    def __init__(self, gen_output_channels, num_resblock=10, vsr_scale=4):
        super(GeneratorF, self).__init__()
        self.encoder = GeneratorFEncoder(num_resblock)
        self.decoder = GeneratorFDecoder(gen_output_channels, vsr_scale)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded, x)
