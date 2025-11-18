import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class MFQE2(nn.Module):
    def __init__(self):
        super(MFQE2, self).__init__()

        # Multi-scale feature extraction layers
        self.conv3_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        self.conv5_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv5_2 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv5_3 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        
        self.conv7_1 = nn.Conv2d(1, 32, kernel_size=7, padding=3)
        self.conv7_2 = nn.Conv2d(1, 32, kernel_size=7, padding=3)
        self.conv7_3 = nn.Conv2d(1, 32, kernel_size=7, padding=3)

        # Dense reconstruction layers
        self.conv1 = nn.Conv2d(32*3*3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32*2, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32*3, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32*4, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, frame1, frame2, frame3):
        # Multi-scale feature extraction
        c3_1 = F.prelu(self.conv3_1(frame1))
        c5_1 = F.prelu(self.conv5_1(frame1))
        c7_1 = F.prelu(self.conv7_1(frame1))
        cc_1 = torch.cat([c3_1, c5_1, c7_1], dim=1)

        c3_2 = F.prelu(self.conv3_2(frame2))
        c5_2 = F.prelu(self.conv5_2(frame2))
        c7_2 = F.prelu(self.conv7_2(frame2))
        cc_2 = torch.cat([c3_2, c5_2, c7_2], dim=1)

        c3_3 = F.prelu(self.conv3_3(frame3))
        c5_3 = F.prelu(self.conv5_3(frame3))
        c7_3 = F.prelu(self.conv7_3(frame3))
        cc_3 = torch.cat([c3_3, c5_3, c7_3], dim=1)

        # Merge
        c_concat = torch.cat([cc_1, cc_2, cc_3], dim=1)

        # Dense reconstruction
        c1 = F.prelu(self.conv1(c_concat))
        c1 = F.batch_norm(c1)

        c2 = F.prelu(self.conv2(c1))
        c2 = F.batch_norm(c2)

        cc2 = torch.cat([c1, c2], dim=1)

        c3 = F.prelu(self.conv3(cc2))
        c3 = F.batch_norm(c3)

        cc3 = torch.cat([c1, c2, c3], dim=1)

        c4 = F.prelu(self.conv4(cc3))
        c4 = F.batch_norm(c4)

        cc4 = torch.cat([c1, c2, c3, c4], dim=1)

        c5 = F.prelu(self.conv5(cc4))
        c5 = F.batch_norm(c5)

        c6 = F.prelu(self.conv6(c5))
        c6 = F.batch_norm(c6)

        # Short connection
        output = c6 + frame2
        return output