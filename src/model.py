#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from gc import collect

class PSPnet(nn.Module):
    def __init__(self):
        super(PSPnet, self).__init__()
        # self.DL = DataLoader()
        self.pool = nn.MaxPool2d(2, 2)
        self.refinements = []

        # Basic downsampling
        self.conv1 = nn.Conv2d(5, 16, 3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=2, dilation=2)
        self.conv5 = nn.Conv2d(128, 512, 3, padding=2, dilation=2)

        self.global_flat_conv = nn.Conv2d(128, 1, 1)
        self.flat_conv = nn.Conv2d(2, 1, 1)
        self.upsample2 = nn.Upsample(scale_factor=(2,2), mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=(4,4), mode="bilinear", align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=(8,8), mode="bilinear", align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=(16,16), mode="bilinear", align_corners=True)
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

        # print("x shape = ", x.shape)
        # creating first bitmap of W*H*64 size 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) 
        x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))


        # print("x shape = ", x.shape)
        # save original feature map for concat at the end
        x0 = x # 1/1 size of original image

        # start creating smaller feature maps for better understanding global context
        x = self.pool(x) # 1/2 size of original image
        x1 = x
        x = self.pool(x) # 1/4 size of original image
        x2 = x
        x = self.pool(x) # 1/8 size of original image
        x3 = x
        x4 = self.pool(x) # 1/16 size of original image



        # Each feature map filter through 1x1 conv layer, which flattens 3. dimension
        x0 = F.relu(self.global_flat_conv(x0))
        x1 = F.relu(self.global_flat_conv(x1))
        x2 = F.relu(self.global_flat_conv(x2))
        x3 = F.relu(self.global_flat_conv(x3))
        x4 = F.relu(self.global_flat_conv(x4))


        # TODO 
        # use refinement maps 
        x4_copy = x4

        
        # Start from smallest map and upsamle to the size of previus layer.
        # Concatenate all layers
        # After every concatenation, flatten 3. dimension
        x3_copy = F.relu(self.flat_conv(torch.cat((x3, self.upsample2(x4)), 1))) # 1/8 size of an image
        x2_copy = F.relu(self.flat_conv(torch.cat((x2, self.upsample2(x3_copy)), 1))) # 1/4 size of an image
        x1_copy = F.relu(self.flat_conv(torch.cat((x1, self.upsample2(x2_copy)), 1))) # 1/2 size of an image
        x0_copy = F.relu(self.flat_conv(torch.cat((x0, self.upsample2(x1_copy)), 1))) # 1/1 size of an image


        # At the end concat once again all layers with individual upsampling
        x0 = torch.cat((x0_copy, 
                        self.upsample2(x1_copy), 
                        self.upsample4(x2_copy), 
                        self.upsample8(x3_copy), 
                        self.upsample16(x4_copy)), 1)
        
        del x0_copy
        del x
        del x1
        del x1_copy
        del x2
        del x2_copy
        del x3
        del x3_copy
        del x4
        del x4_copy
        collect()

        # Apply last convolution layer to make 2D image
        return F.relu(self.final_conv(x0))


    def add_refinement_map(self, img, x):
        """Funtion for adding additional positive clicks by user
        Funtion will store x into model and trigger foward funtion, which then returns updated segmentation image

        Args:
        -----
            img (img): original image (including bounding box & 1 positive click feature maps)
            x (additional feature map): same size feature map of another positive click
        """
        self.refinements.append(x)
        return self.forward(img)

    def load_model(self, filename):
        pass

    def save_model(self, filename):
        pass



