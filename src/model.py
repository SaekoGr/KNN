#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PSPnet(nn.Module):
    def __init__(self):
        super(PSPnet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.refinment_maps = None

        # Basic downsampling
        self.conv1 = nn.Conv2d(5, 16, 3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1, dilation=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, dilation=1)
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1, dilation=1)
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1, dilation=1)


        self.flat_conv = nn.Conv2d(513, 1, 1)
        self.adjust_conv = nn.Conv2d(513, 512, 1)
        self.same_size_conv = nn.Conv2d(512*2, 512, 1)

        self.upsample2 = nn.Upsample(scale_factor=(2,2), mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=(4,4), mode="bilinear", align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=(8,8), mode="bilinear", align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=(16,16), mode="bilinear", align_corners=True)

        self.full_pool16 = nn.MaxPool2d((16,16), 16)
        self.final_conv1_wrm = nn.Conv2d(512*5+1, 1024, 3)
        self.final_conv1_worm = nn.Conv2d(512*5, 1024, 3, padding=1)
        self.final_conv2 = nn.Conv2d(1024, 256, 3, padding=1)
        self.final_conv3 = nn.Conv2d(256, 1, 3, padding=1)


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
        # creating first bitmap of W*H size
        x = F.relu(self.conv1(x)) # W*H*16
        x = F.relu(self.conv2(x)) # W*H*32
        x = F.relu(self.conv3(x)) # W*H*64
        x = F.relu(self.conv4(x)) # W*H*128
        x = F.relu(self.conv5(x)) # W*H*256



        # save original feature map for concat at the end
        x0 = F.relu(self.conv6(x)) # 1/1 size of original image

        # start creating smaller feature maps for better understanding global context
        # all maps has 3. dimension of 512
        x = self.pool(F.relu(self.conv6(x))) # 1/2 size of original image
        x1 = x
        x = self.pool(F.relu(self.conv7(x))) # 1/4 size of original image
        x2 = x
        x = self.pool(F.relu(self.conv7(x))) # 1/8 size of original image
        x3 = x
        x4 = self.pool(F.relu(self.conv7(x))) # 1/16 size of original image


        # # TODO 
        # # use refinement maps 
        # save mid result for adding refinment maps
        # self.mid_res = [x, x0, x1, x2, x3, x4]

        # x5 includes additive click from user
        if type(self.refinment_maps) != type(None):
            x5 = F.relu(self.flat_conv(torch.cat((x4, self.refinment_maps), 1)))
            x4 = F.relu(self.adjust_conv(torch.cat((x4, x5))))

        # Start from smallest map and upsamle to the size of previus layer.
        # Concatenate all layers
        # After every concatenation
        x3 = F.relu(self.same_size_conv(torch.cat((x3, self.upsample2(x4)), 1))) # 1/8 size of an image
        x2 = F.relu(self.same_size_conv(torch.cat((x2, self.upsample2(x3)), 1))) # 1/4 size of an image
        x1 = F.relu(self.same_size_conv(torch.cat((x1, self.upsample2(x2)), 1))) # 1/2 size of an image
        x0 = F.relu(self.same_size_conv(torch.cat((x0, self.upsample2(x1)), 1))) # 1/1 size of an image



        # At the end concat once again all layers with individual upsampling
        x = torch.cat((x0, 
                        self.upsample2(x1), 
                        self.upsample4(x2), 
                        self.upsample8(x3), 
                        self.upsample16(x4),
                        ), 1)
                
        if type(self.refinment_maps) != type(None):
            x = torch.cat((x, self.upsample16(x5)), 1)
            self.refinment_maps = None
            x = F.relu(self.final_conv_wrm(x))
        else:
            x = F.relu(self.final_conv1_worm(x))


        x = F.relu(self.final_conv2(x))
        y = F.relu(self.final_conv3(x))
        
        del x
        del x0
        del x1
        del x2
        del x3
        del x4
        torch.cuda.empty_cache()
        return y


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


    def add_refinement_map_train(self, refinment_maps):
        """Funtion for adding additional positive clicks by user
        Funtion will store x into model and trigger foward funtion, which then returns updated segmentation image

        Args:
        -----
            img (img): original image (including bounding box & 1 positive click feature maps)
            x (additional feature map): same size feature map of another positive click
        """

        self.refinment_maps = self.full_pool16(refinment_maps)

    def load_model(self, filename):
        pass

    def save_model(self, filename):
        pass



