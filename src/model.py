#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

class IOGnet(nn.Module):
    def __init__(self):
        super(IOGnet, self).__init__()
        self.pool2 = nn.AvgPool2d(2, 2)
        self.upsample2 = nn.Upsample(scale_factor=(2,2), mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=(4,4), mode="bilinear", align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=(8,8), mode="bilinear", align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=(16,16), mode="bilinear", align_corners=True)
        self.bn1 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(1024)
        self.bn3 = nn.BatchNorm2d(384)


        self.enter_conv1 = nn.Conv2d(5, 64, 3, padding=1)
        self.enter_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.enter_conv3 = nn.Conv2d(64, 64, 1)

        self.cors_enc_conv1_1 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.cors_enc_conv1_2 = nn.Conv2d(128, 128, 1)
        self.cors_enc_conv1_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.cors_enc_conv1_4 = nn.Conv2d(128, 128, 1)
        self.cors_enc_conv1_5 = nn.Conv2d(256, 128, 3, padding=1)

        self.cors_enc_conv1_6 = nn.Conv2d(128, 128, 1)
        self.cors_enc_conv1_7 = nn.Conv2d(128, 128, 3, padding=1)
        self.cors_enc_conv1_8 = nn.Conv2d(128, 128, 1)
        self.cors_enc_conv1_9 = nn.Conv2d(256, 128, 3, padding=1)

        self.cors_enc_conv1_10 = nn.Conv2d(128, 128, 1)
        self.cors_enc_conv1_11 = nn.Conv2d(128, 128, 3, padding=1)
        self.cors_enc_conv1_12 = nn.Conv2d(128, 128, 1)


        self.cors_enc_conv2_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.cors_enc_conv2_2 = nn.Conv2d(256, 256, 1)
        self.cors_enc_conv2_3 = nn.Conv2d(256, 256, 1)
        self.cors_enc_conv2_4 = nn.Conv2d(512, 256, 3, padding=1)

        self.cors_enc_conv2_5 = nn.Conv2d(256, 256, 1)
        self.cors_enc_conv2_6 = nn.Conv2d(256, 256, 3, padding=1)
        self.cors_enc_conv2_7 = nn.Conv2d(256, 256, 1)

        self.cors_enc_conv3_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.cors_enc_conv3_2 = nn.Conv2d(512, 512, 1)
        self.cors_enc_conv3_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.cors_enc_conv4_1 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.cors_enc_conv4_2 = nn.Conv2d(1024, 1024, 1)
        self.cors_dec_conv4_3 = nn.Conv2d(1024, 512, 3, padding=1)
        
        self.cors_dec_conv3_1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.cors_dec_conv3_2 = nn.Conv2d(512, 512, 1)
        self.cors_dec_conv3_3 = nn.Conv2d(512, 256, 3, padding=1)
        self.cors_dec_conv3_4 = nn.Conv2d(512, 256, 3, padding=1)

        self.cors_dec_conv2_1 = nn.Conv2d(256, 256, 1)
        self.cors_dec_conv2_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.cors_dec_conv2_3 = nn.Conv2d(256, 256, 1)
        self.cors_dec_conv2_4 = nn.Conv2d(256, 128, 3, padding=1)
        self.cors_dec_conv2_5 = nn.Conv2d(256, 128, 3, padding=1)

        self.cors_dec_conv1_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.cors_dec_conv1_2 = nn.Conv2d(128, 128, 1)
        self.cors_dec_conv1_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.cors_dec_conv1_4 = nn.Conv2d(128, 128, 1)
        self.cors_dec_conv1_5 = nn.Conv2d(128, 64, 3, padding=1)
        self.cors_dec_conv1_6 = nn.Conv2d(128, 64, 3, padding=1)

        self.cors_dec_conv0_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.cors_dec_conv0_2 = nn.Conv2d(64, 64, 1)


        # FINE NET
        self.fine_work4_1 = nn.Conv2d(512, 256, 3, padding=1)
        self.fine_work4_2 = nn.Conv2d(256, 128, 1)

        self.fine_work3_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.fine_work3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.fine_work3_3 = nn.Conv2d(128, 64, 1)

        self.fine_work2_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.fine_work2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.fine_work2_3 = nn.Conv2d(64, 64, 1)

        self.fine_work1_1 = nn.Conv2d(64, 64, 3, padding=1)

        self.fine_final1 = nn.Conv2d(384, 384, 3, padding=1)
        self.fine_final2 = nn.Conv2d(384, 256, 3, padding=1)
        self.fine_final3 = nn.Conv2d(256, 128, 3, padding=1)
        self.fine_final4 = nn.Conv2d(128, 32, 3, padding=1)
        self.fine_final5 = nn.Conv2d(32, 1, 1)







    def forward(self, x0):

        # Basic work on original image
        x0 = F.relu(self.enter_conv1(x0)) # 64
        x0 = F.relu(self.enter_conv2(x0)) # 64
        x0 = F.relu(self.enter_conv3(x0)) # 64
        x1 = self.pool2(x0) # 1/2 1/2 64


        x1 =  F.relu(self.cors_enc_conv1_1(x1)) # 1/2 1/2 128

        x_copy = F.relu(self.cors_enc_conv1_2(x1)) # 1/2 1/2 128
        x1 = F.relu(self.cors_enc_conv1_3(x_copy)) # 1/2 1/2 128
        x1 = F.relu(self.cors_enc_conv1_4(x1)) # 1/2 1/2 128
        x1 = F.relu(self.cors_enc_conv1_5(torch.cat((x1, x_copy), 1)))  # 1/2 1/2 128

        x_copy = F.relu(self.cors_enc_conv1_6(x1)) # 1/2 1/2 128
        x1 = F.relu(self.cors_enc_conv1_7(x_copy)) # 1/2 1/2 128
        x1 = F.relu(self.cors_enc_conv1_8(x1)) # 1/2 1/2 128
        x1 = F.relu(self.cors_enc_conv1_9(torch.cat((x1, x_copy), 1)))  # 1/2 1/2 128
    
        x_copy = F.relu(self.cors_enc_conv1_10(x1)) # 1/2 1/2 128
        x1 = F.relu(self.cors_enc_conv1_11(x_copy)) # 1/2 1/2 128
        x1 = F.relu(self.cors_enc_conv1_12(x1)) # 1/2 1/2 128
        x2 = torch.cat((x1, x_copy), 1) # 1/2 1/2 256

        # Second layer of encoder
        x2 = self.pool2(x2) # 1/4 1/4 256
        
        x_copy = F.relu(self.cors_enc_conv2_1(x2))
        x2 = F.relu(self.cors_enc_conv2_2(x_copy))
        x2 = F.relu(self.cors_enc_conv2_3(x2))
        x2 = F.relu(self.cors_enc_conv2_4(torch.cat((x2, x_copy), 1)))  # 1/4 1/4 256

        x_copy = F.relu(self.cors_enc_conv2_5(x2))
        x2 = F.relu(self.cors_enc_conv2_6(x_copy))
        x2 = F.relu(self.cors_enc_conv2_7(x2))
        x3 = torch.cat((x2, x_copy), 1)  # 1/4 1/4 512

        # Third layer of encoder
        x3 = self.pool2(x3) # 1/8 1/8 512

        x_copy = F.relu(self.cors_enc_conv3_1(x3))
        x3 = F.relu(self.cors_enc_conv3_2(x_copy))
        x3 = F.relu(self.cors_enc_conv3_3(x3))
        x4 = torch.cat((x3, x_copy), 1)  # 1/8 1/8 1024
        del x_copy

        # Fourth layer of encoder (last)
        x4 = self.pool2(x4) # 1/16 1/16 1024
        x4 = self.bn1(x4)
        x4 = F.relu(self.cors_enc_conv4_1(x4))
        x4 = self.bn2(x4)
        x4 = F.relu(self.cors_enc_conv4_2(x4))

        #
        # Start of decoder
        #

        # Fourth layer 1/16
        x4 = F.relu(self.cors_dec_conv4_3(x4)) # 512

        # Third layer 1/8
        x3 = F.relu(self.cors_dec_conv3_1(
            torch.cat(
                (
                    self.upsample2(x4),
                    x3
                ), 1
            )
        )) # 512

        x3 = F.relu(self.cors_dec_conv3_2(x3))
        x3 = F.relu(self.cors_dec_conv3_3(x3)) # 256

        # Second layer 1/4
        x2 = F.relu(self.cors_dec_conv3_4(
            torch.cat(
                (
                    self.upsample2(x3),
                    x2
                ), 1
            )
        )) # 256


        x2 = F.relu(self.cors_dec_conv2_1(x2))
        x2 = F.relu(self.cors_dec_conv2_2(x2))
        x2 = F.relu(self.cors_dec_conv2_3(x2))
        x2 = F.relu(self.cors_dec_conv2_4(x2))

        # First layer 1/2
        x1 = F.relu(self.cors_dec_conv2_5(
            torch.cat(
                (
                    self.upsample2(x2),
                    x1
                ), 1
            )
        )) # 128

        x1 = F.relu(self.cors_dec_conv1_1(x1))
        x1 = F.relu(self.cors_dec_conv1_2(x1))
        x1 = F.relu(self.cors_dec_conv1_3(x1))
        x1 = F.relu(self.cors_dec_conv1_4(x1))
        x1 = F.relu(self.cors_dec_conv1_5(x1))

        # Original size
        x0 = F.relu(self.cors_dec_conv1_6(
            torch.cat(
                (
                    self.upsample2(x1),
                    x0
                ), 1
            )
        )) # 64

        x0 = F.relu(self.cors_dec_conv0_1(x0))
        x0 = F.relu(self.cors_dec_conv0_2(x0)) # 64


        # FINE NET
        x4 = F.relu(self.fine_work4_1(self.upsample4(x4)))
        x4 = F.relu(self.fine_work4_2(self.upsample4(x4)))

        x3 = F.relu(self.fine_work3_1(self.upsample4(x3)))
        x3 = F.relu(self.fine_work3_2(x3))
        x3 = F.relu(self.fine_work3_3(self.upsample2(x3)))

        x2 = F.relu(self.fine_work2_1(self.upsample2(x2)))
        x2 = F.relu(self.fine_work2_2(x2))
        x2 = F.relu(self.fine_work2_3(self.upsample2(x2)))

        x1 = F.relu(self.fine_work1_1(self.upsample2(x1)))

        x0 = F.relu(self.fine_final1(torch.cat(
                (
                    x0, #64
                    x1, #64
                    x2, #64
                    x3, #64
                    x4, #128
                ), 1
            )
        ))
        x0 = self.bn3(x0)
        x0 = F.relu(self.fine_final2(x0))
        x0 = F.relu(self.fine_final3(x0))
        x0 = F.relu(self.fine_final4(x0))
        return torch.sigmoid(self.fine_final5(x0))