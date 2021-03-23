#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import DataLoader

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.DL = DataLoader()
        self.pool = nn.MaxPool2d(2, 2)

        # Basic downsampling
        self.conv1 = nn.Conv2d(5, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, dilation=2)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1, dilation=2)

        self.global_flat_conv = nn.Conv2d(128, 1, 1)
        self.flat_conv = nn.Conv2d(2, 1, 1)
        self.upsample2 = nn.Upsample(scale_factor=(2,2,1), mode="bilinear")
        self.upsample4 = nn.Upsample(scale_factor=(4,4,1), mode="bilinear")
        self.upsample8 = nn.Upsample(scale_factor=(8,8,1), mode="bilinear")
        self.upsample16 = nn.Upsample(scale_factor=(16,16,1), mode="bilinear")
        self.final_conv = nn.Conv2d(5, 1, 3, padding=1)


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

        # creating first bitmap of W*H*64 size 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) 
        x = F.relu(self.conv4(x)) 



        # save original feature map for concat at the end
        x0 = deepcopy(x) # 1/1 size of original image

        # start creating smaller feature maps for better understanding global context
        x = self.pool(x) # 1/2 size of original image
        x1 = deepcopy(x)
        x = self.pool(x) # 1/4 size of original image
        x2 = deepcopy(x)
        x = self.pool(x) # 1/8 size of original image
        x3 = deepcopy(x)
        x4 = self.pool(x) # 1/16 size of original image



        # Each feature map filter through 1x1 conv layer, which flattens 3. dimension
        x0 = self.global_flat_conv(F.relu(x0))
        x1 = self.global_flat_conv(F.relu(x1))
        x2 = self.global_flat_conv(F.relu(x2))
        x3 = self.global_flat_conv(F.relu(x3))
        x4 = self.global_flat_conv(F.relu(x4))


        # TODO 
        # use refinement maps 
        x4_copy = deepcopy(x4)

        
        # Start from smallest map and upsamle to the size of previus layer.
        # Concatenate all layers
        # After every concatenation, flatten 3. dimension
        x3_copy = F.relu(self.flat_conv(torch.cat((x3, self.upsample2(x4)), 0))) # 1/8 size of an image
        x2_copy = F.relu(self.flat_conv(torch.cat((x2, self.upsample2(x3_copy)), 0))) # 1/4 size of an image
        x1_copy = F.relu(self.flat_conv(torch.cat((x1, self.upsample2(x2_copy)), 0))) # 1/2 size of an image
        x0_copy = F.relu(self.flat_conv(torch.cat((x0, self.upsample2(x1_copy)), 0))) # 1/1 size of an image


        # At the end concat once again all layers with individual upsampling
        x0 = torch.cat((x0_copy, 
                        self.upsample2(x1_copy), 
                        self.upsample4(x2_copy), 
                        self.upsample8(x3_copy), 
                        self.upsample16(x4_copy)))
        
        # Apply last convolution layer to make 2D image
        y = F.relu(self.final_conv(x0))
        return y


    def add_refinement_map(x):
        ...

    def load_model(self, filename):
        pass

    def save_model(self, filename):
        pass



