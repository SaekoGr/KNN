#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

class IOGnet(nn.Module):
    def __init__(self):
        super(IOGnet, self).__init__()
        self.pool2 = nn.MaxPool2d(2, 2)


        # Basic downsampling
        self.conv_enter_enc1 = nn.Conv2d(5, 32, 3, padding=1, dilation=1)
        self.conv_enter_enc2 = nn.Conv2d(32, 64, 3, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv1_enc = nn.Conv2d(64, 128, 3, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_enc = nn.Conv2d(128, 256, 3, padding=1, dilation=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3_enc = nn.Conv2d(256, 516, 3, padding=1, dilation=1)
        self.bn4 = nn.BatchNorm2d(516)
        self.conv4_enc = nn.Conv2d(516, 768, 3, padding=1, dilation=1)
        self.bn5 = nn.BatchNorm2d(768)

        self.conv1_dec = nn.Conv2d(768+516, 768, 3, padding=1, dilation=1)
        self.bn6 = nn.BatchNorm2d(768)
        self.conv2_dec = nn.Conv2d(768+256, 516, 3, padding=1, dilation=1)
        self.bn7 = nn.BatchNorm2d(516)
        self.conv3_dec = nn.Conv2d(516+128, 256, 3, padding=1, dilation=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv4_dec = nn.Conv2d(256+64, 128, 3, padding=1, dilation=1)
        self.bn9 = nn.BatchNorm2d(128)


        self.upsample2 = nn.Upsample(scale_factor=(2,2), mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=(4,4), mode="bilinear", align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=(8,8), mode="bilinear", align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=(16,16), mode="bilinear", align_corners=True)


        self.final_conv1 = nn.Conv2d(768, 256, 3, padding=1, dilation=1)
        self.final_conv2 = nn.Conv2d(256, 64, 3, padding=1, dilation=1)
        self.final_conv3 = nn.Conv2d(768, 256, 3, padding=1, dilation=1)
        self.final_conv4 = nn.Conv2d(256, 64, 3, padding=1, dilation=1)
        self.final_conv5 = nn.Conv2d(516, 256, 3, padding=1, dilation=1)
        self.final_conv6 = nn.Conv2d(256, 64, 3, padding=1, dilation=1)
        self.final_conv7 = nn.Conv2d(256, 64, 3, padding=1, dilation=1)
        self.final_conv8 = nn.Conv2d(384, 516, 3, padding=1, dilation=1)
        self.bn10 = nn.BatchNorm2d(516)
        self.final_conv9 = nn.Conv2d(516, 128, 3, padding=1, dilation=1)
        self.bn11 = nn.BatchNorm2d(128)
        self.final_conv10 = nn.Conv2d(128, 32, 3, padding=1, dilation=1)
        self.final_conv11 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        """Function for evaluating input data.
        Be ware that image (x) must have concatenated feature maps (clicks & bounding box) to the image

        Args:
        -----
            x (input image): Size must be W*H*5 (5 = RGB + (bounding box map) + (one positive click map))

        Returns:
        --------
            y: segmented image of W*H*1 (black & white)
        """

        # Encoder of CoarseNet
        x = F.relu(self.conv_enter_enc1(x)) # W*H*32 1/1 size
        x = F.relu(self.conv_enter_enc2(x)) # W*H*64 1/1 size
        x = self.bn1(x)

        x0 = F.relu(self.pool2(self.conv1_enc(x))) # W*H*128 1/2 size
        x0 = self.bn2(x0)
        x1 = F.relu(self.pool2(self.conv2_enc(x0))) # W*H*256 1/4 size
        x1 = self.bn3(x1)
        x2 = F.relu(self.pool2(self.conv3_enc(x1))) # W*H*516 1/8 size
        x2 = self.bn4(x2)
        x3 = F.relu(self.pool2(self.conv4_enc(x2))) # W*H*768 1/16 size
        x3 = self.bn5(x3)

        # Decoder of CoarseNet
        x2 = F.relu(self.conv1_dec(torch.cat((x2, self.upsample2(x3)), 1))) # W/8 * H/8 * 768
        x2 = self.bn6(x2)
        x1 = F.relu(self.conv2_dec(torch.cat((x1, self.upsample2(x2)), 1))) # W/4 * H/4 * 516
        x1 = self.bn7(x1)
        x0 = F.relu(self.conv3_dec(torch.cat((x0, self.upsample2(x1)), 1))) # W/2 * H/2 * 256
        x0 = self.bn8(x0)
        x = F.relu(self.conv4_dec(torch.cat((x, self.upsample2(x0)), 1))) # W * H * 128
        x = self.bn9(x)

        # FineNet
        # Scaling deepest layer
        x3 = F.relu(
            self.final_conv1(
                self.upsample4(x3)
            )
        )
        x3 = F.relu(
            self.final_conv2(
                self.upsample4(x3)
            )
        )

        # Scaling next layer
        x2 = F.relu(
            self.final_conv3(
                self.upsample4(x2)
            )
        )
        x2 = F.relu(
            self.final_conv4(
                self.upsample2(x2)
            )
        )

        # Scaling next layer
        x1 = F.relu(
            self.final_conv5(
                self.upsample2(x1)
            )
        )
        x1 = F.relu(
            self.final_conv6(
                self.upsample2(x1)
            )
        )

        # Scaling next layer
        x0 = F.relu(
            self.final_conv7(
                self.upsample2(x0)
            )
        )

        x = F.relu(self.final_conv8(
            torch.cat(
                (
                    x,
                    x0,
                    x1,
                    x2,
                    x3
                ),1
            )
        ))
        x = self.bn10(x)
        x = F.relu(self.final_conv9(x))
        x = self.bn11(x)
        x = F.relu(self.final_conv10(x))
        x = torch.sigmoid(self.final_conv11(x))
        return x