#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PSPnet(nn.Module):
    def __init__(self):
        super(PSPnet, self).__init__()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool16 = nn.MaxPool2d(16, 16)
        self.refinment_maps = None

        # Basic downsampling
        self.conv_enter_enc = nn.Conv2d(5, 32, 3, padding=2, dilation=2)
        self.conv1_enc = nn.Conv2d(32, 64, 3, padding=2, dilation=2)
        self.conv2_enc = nn.Conv2d(64, 128, 3, padding=2, dilation=2)
        self.conv3_enc = nn.Conv2d(128, 256, 3, padding=2, dilation=2)
        self.conv4_enc = nn.Conv2d(256, 512, 3, padding=1, dilation=1)

        self.conv_ref1 = nn.Conv2d(513, 256, 3, padding=1, dilation=1)
        self.conv_ref2 = nn.Conv2d(256, 64, 3, padding=1, dilation=1)
        self.conv_adjust_ref1 = nn.Conv2d(576, 512, 3, padding=1)
        self.conv_adjust_ref2 = nn.Conv2d(1024, 512, 3, padding=1)

        self.conv1_dec = nn.Conv2d(768, 256, 3, padding=1, dilation=1)
        self.conv2_dec = nn.Conv2d(384, 128, 3, padding=1, dilation=1)
        self.conv3_dec = nn.Conv2d(192, 64, 3, padding=1, dilation=1)
        self.conv4_dec = nn.Conv2d(96, 32, 3, padding=1, dilation=1)


        self.upsample2 = nn.Upsample(scale_factor=(2,2), mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=(4,4), mode="bilinear", align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=(8,8), mode="bilinear", align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=(16,16), mode="bilinear", align_corners=True)


        self.final_conv1 = nn.Conv2d(992, 512, 3, padding=1, dilation=1)
        self.final_conv2 = nn.Conv2d(512, 64, 1)
        self.final_conv3 = nn.Conv2d(64, 1, 1)





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
        x = F.relu(self.conv_enter_enc(x)) # W*H*32 1/1 size
        x0 = F.relu(self.pool2(self.conv1_enc(x))) # W*H*64 1/2 size
        x1 = F.relu(self.pool2(self.conv2_enc(x0))) # W*H*128 1/4 size
        x2 = F.relu(self.pool2(self.conv3_enc(x1))) # W*H*256 1/8 size
        x3 = F.relu(self.pool2(self.conv4_enc(x2))) # W*H*512 1/16 size


        # refinement maps module
        if type(self.refinment_maps) != type(None):
            refs = self.pool16(F.relu(nn.Conv2d(self.refinment_maps.shape[0], 1, 1))) # W/16 * H/16 * 1
            x4 = F.relu(self.conv_ref1(torch.cat((x3, refs), 1))) # W/16 * H/16 * 256
            x4 = F.relu(self.conv_ref2(x4)) # W/16 * H/16 * 64
            x3_copy = F.relu(self.conv_adjust_ref1(torch.cat((x3, x4), 1))) # W/16 * H/16 * 512
            x3 = F.relu(self.conv_adjust_ref2(torch.cat((x3, x3_copy), 1))) # W/16 * H/16 * 512


        # Decoder of CoarseNet
        x2 = F.relu(self.conv1_dec(torch.cat((x2, self.upsample2(x3)), 1))) # W/8 * H/8 * 256
        x1 = F.relu(self.conv2_dec(torch.cat((x1, self.upsample2(x2)), 1))) # W/4 * H/4 * 128
        x0 = F.relu(self.conv3_dec(torch.cat((x0, self.upsample2(x1)), 1))) # W/2 * H/2 * 64
        x = F.relu(self.conv4_dec(torch.cat((x, self.upsample2(x0)), 1))) # W * H * 32

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
        y = F.relu(self.final_conv2(x))
        y = F.relu(self.final_conv3(y))

        del x0
        del x1
        del x2
        del x3
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



