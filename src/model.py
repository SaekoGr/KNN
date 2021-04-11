#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

class IOGnet(nn.Module):
    def __init__(self):
        super(IOGnet, self).__init__()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool16 = nn.MaxPool2d(16, 16)
        self.refinment_maps = None

        # Basic downsampling
        self.conv_enter_enc1 = nn.Conv2d(5, 16, 3, padding=2, dilation=2)
        self.conv_enter_enc2 = nn.Conv2d(16, 64, 3, padding=2, dilation=2)

        self.conv1_enc = nn.Conv2d(64, 92, 3, padding=2, dilation=2)
        self.conv2_enc = nn.Conv2d(92, 156, 3, padding=2, dilation=2)
        self.conv3_enc = nn.Conv2d(156, 256, 3, padding=2, dilation=2)
        self.conv4_enc = nn.Conv2d(256, 512, 3, padding=1, dilation=1)

        self.conv1_dec = nn.Conv2d(512+256, 256, 3, padding=1, dilation=1)
        self.conv2_dec = nn.Conv2d(256+156, 128, 3, padding=1, dilation=1)
        self.conv3_dec = nn.Conv2d(128+92, 64, 3, padding=1, dilation=1)
        self.conv4_dec = nn.Conv2d(64+64, 32, 3, padding=1, dilation=1)


        self.upsample2 = nn.Upsample(scale_factor=(2,2), mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=(4,4), mode="bilinear", align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=(8,8), mode="bilinear", align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=(16,16), mode="bilinear", align_corners=True)


        self.final_conv1 = nn.Conv2d(32+64+128+256+512, 512, 3, padding=1, dilation=1)
        self.final_conv2 = nn.Conv2d(512, 256, 3, padding=1, dilation=1)
        self.final_conv3 = nn.Conv2d(256, 64, 3, padding=1, dilation=1)
        self.final_conv4 = nn.Conv2d(64, 1, 1)

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
        x = F.relu(self.conv_enter_enc1(x)) # W*H*8 1/1 size
        x = F.relu(self.conv_enter_enc2(x)) # W*H*8 1/1 size

        x0 = F.relu(self.pool2(self.conv1_enc(x))) # W*H*16 1/2 size
        x1 = F.relu(self.pool2(self.conv2_enc(x0))) # W*H*32 1/4 size
        x2 = F.relu(self.pool2(self.conv3_enc(x1))) # W*H*64 1/8 size
        x3 = F.relu(self.pool2(self.conv4_enc(x2))) # W*H*128 1/16 size

        # Decoder of CoarseNet
        x2 = F.relu(self.conv1_dec(torch.cat((x2, self.upsample2(x3)), 1))) # W/8 * H/8 * 64
        x1 = F.relu(self.conv2_dec(torch.cat((x1, self.upsample2(x2)), 1))) # W/4 * H/4 * 32
        x0 = F.relu(self.conv3_dec(torch.cat((x0, self.upsample2(x1)), 1))) # W/2 * H/2 * 16
        x = F.relu(self.conv4_dec(torch.cat((x, self.upsample2(x0)), 1))) # W * H * 8

        # FineNet
        x = F.relu(self.final_conv1(
            torch.cat(
                (
                    x,
                    self.upsample2(x0),
                    self.upsample4(x1),
                    self.upsample8(x2),
                    self.upsample16(x3)
                ),1
            )
        ))
        x = F.relu(self.final_conv2(x))
        x = F.relu(self.final_conv3(x))
        x = torch.sigmoid(self.final_conv4(x))
        return x


    def add_refinement_map(self, img, x):
        """Funtion for adding additional positive clicks by user
        Funtion will store x into model and trigger foward funtion, which then returns updated segmentation image

        Args:
        -----
            img (img): original image (including bounding box & 1 positive click feature maps)
            x (additional feature map): same size feature map of another positive click
        """
        ...
        # self.refinements.append(x)
        # return self.forward(img)


    def add_refinement_map_train(self, refinment_maps):
        """Funtion for adding additional positive clicks by user
        Funtion will store x into model and trigger foward funtion, which then returns updated segmentation image

        Args:
        -----
            img (img): original image (including bounding box & 1 positive click feature maps)
            x (additional feature map): same size feature map of another positive click
        """

        self.refinment_maps = refinment_maps

    def load_model(self, filename):
        pass

    def save_model(self, filename):
        pass



