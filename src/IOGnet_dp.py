#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout2d

def create_conv_block(input, f1, f2, f3):
    layers = []
    layers.append(nn.Conv2d(input, f1, 1))
    layers.append(nn.Conv2d(f1, f2, 3, padding=1))
    layers.append(nn.Conv2d(f2, f3, 1))
    layers.append(nn.BatchNorm2d(f3, track_running_stats=False))
    layers.append(nn.Conv2d(input, f3, 1))
    layers.append(nn.BatchNorm2d(f3, track_running_stats=False))
    layers.append(nn.Conv2d(2*f3, f3, 3, padding=1))

    return nn.ModuleList(layers) 


def conv_block(x, layers, relu):
    x_shortcut = x
    x = relu(layers[0](x))
    x = relu(layers[1](x))
    x = layers[2](x)
    x = layers[3](x) # batch norm.
    x_shortcut = layers[5](layers[4](x_shortcut))
    x = relu(
            layers[6](
                torch.cat([
                    x,
                    x_shortcut
                ], 1)
            )
        )
    return x

def create_id_block(input, f1, f2, f3):
    layers = [
        nn.Conv2d(input, f1, 1),
        nn.Dropout2d(0.2),
        nn.Conv2d(f1, f2, 3, padding=1),
        nn.Conv2d(f2, f3, 1),
        nn.BatchNorm2d(f3, track_running_stats=False),
        nn.Conv2d(2*f3, f3, 3, padding=1),
    ]

    return nn.ModuleList(layers)

def id_block(x, layers, relu):
    x_shortcut = x
    x = relu(layers[0](x))
    x = layers[1](x)
    x = relu(layers[2](x))
    x = layers[3](x)
    x = layers[4](x)
    x = relu(
        torch.cat([
            x,
            x_shortcut
        ], 1)
    )
    x = layers[5](x)

    return x


def create_decoder_block(x_sm_size, x_lg_size, upsamle):
    layers = [
        upsamle,
        nn.Conv2d(x_sm_size + x_lg_size, x_sm_size, 3, padding=1),
        nn.BatchNorm2d(x_sm_size, track_running_stats=False),

        nn.Conv2d(x_sm_size, x_sm_size, 1),
        nn.Dropout2d(0.15),

        nn.Conv2d(x_sm_size, x_sm_size//2, 3, padding=1),
        nn.BatchNorm2d(x_sm_size//2, track_running_stats=False),

        nn.Conv2d(x_sm_size//2 + x_sm_size, x_sm_size//2, 3, padding=1),
        nn.BatchNorm2d(x_sm_size//2, track_running_stats=False),
    ]

    return nn.ModuleList(layers)

def decoder_block(x_sm, x_lg, layers, relu):
    x_sm = relu(layers[2](layers[1](torch.cat(
        [layers[0](x_sm), x_lg]
        , 1
    ))))
    x_lg = relu(layers[3](x_sm))
    x_lg = layers[4](x_lg)
    x_lg = relu(layers[6](layers[5](x_lg)))

    x_sm = relu(layers[8](layers[7](torch.cat(
        [x_sm, x_lg],
        1
    ))))

    return x_sm


def create_finenet_block(upsamle_rates, input, f1, f2):
    layers = []
    for i, rate in enumerate(upsamle_rates):
        if i > 0:
            input = f2
            f1 //=2
            f2 //=2
        layers.append(nn.Upsample(scale_factor=(rate, rate), mode="bilinear", align_corners=True))
        layers.append(nn.Conv2d(input, f1, 3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(f1, f2, 1))
        layers.append(nn.BatchNorm2d(f2, track_running_stats=False))
        layers.append(nn.ReLU())
    
    return nn.ModuleList(layers)
    

def finenet_block(x, layers):
    for l in layers:
        x = l(x)
    
    return x




class IOGnet(nn.Module):
    def __init__(self):
        super(IOGnet, self).__init__()
        self.maxpool = nn.MaxPool2d((2,2))
        # self.avgpool = nn.AvgPool2d((2,2))


        self.upsample2 = nn.Upsample(scale_factor=(2,2), mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=(4,4), mode="bilinear", align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=(8,8), mode="bilinear", align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=(16,16), mode="bilinear", align_corners=True)


        # Stage 1 Encoder
        self.zero_pad = nn.ZeroPad2d((3, 3))
        self.coars_enc_conv_s1 = nn.Conv2d(5, 64, 7, padding=(3,3))
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)

        # Stage 2 Encoder
        # Conv_block
        self.conv_block2 = create_conv_block(64, 64, 64, 128)

        # Id_block 1
        self.id_block2_1 = create_id_block(128, 64, 64, 128)

        # Stage 3 Encoder
        # Conv_block
        self.conv_block3 = create_conv_block(128, 128, 128, 256)

        # Id_block 1, 2, 3
        self.id_block3_1 = create_id_block(256, 128, 128, 256)
        self.id_block3_2 = create_id_block(256, 128, 128, 256)
        self.id_block3_3 = create_id_block(256, 128, 128, 256)

        # Stage 4 Encoder
        # Conv_block
        self.conv_block4 = create_conv_block(256, 128, 128, 512)

        # Id_block 1, 2, 3, 4, 5
        self.id_block4_1 = create_id_block(512, 256, 256, 512)
        self.id_block4_2 = create_id_block(512, 256, 256, 512)
        self.id_block4_3 = create_id_block(512, 256, 256, 512)
        self.id_block4_4 = create_id_block(512, 256, 256, 512)
        self.id_block4_5 = create_id_block(512, 256, 256, 512)

        # Stage 4 Encoder
        # Conv_block
        self.conv_block5 = create_conv_block(512, 256, 256, 1024)

        # Id block 1,2
        self.id_block5_1 = create_id_block(1024, 512, 512, 1024)
        self.id_block5_2 = create_id_block(1024, 512, 512, 1024)

        #
        # START OF DECODER
        #

        self.decoder_block1 = create_decoder_block(1024, 512, self.upsample2) # 1/8
        self.decoder_block2 = create_decoder_block(512, 256, self.upsample2) # 1/4
        self.decoder_block3 = create_decoder_block(256, 128, self.upsample2) # 1/2
        self.decoder_block4 = create_decoder_block(128, 64, self.upsample2) # 1

        #
        # START OF FINENET
        #

        self.fine_block1 = create_finenet_block([4, 4], 1024, 512, 512)
        self.fine_block2 = create_finenet_block([4, 2], 512, 256, 256)
        self.fine_block3 = create_finenet_block([2, 2], 256, 128, 128)
        self.fine_block4 = create_finenet_block([2], 128, 64, 64)
        self.fine_block5 = create_finenet_block([1], 64, 32, 32)

        self.final_conv1 = nn.Conv2d(544, 256, 3, padding=1)
        self.final_bn1 = nn.BatchNorm2d(256, track_running_stats=False)
        self.final_conv2 = nn.Conv2d(256, 64, 3, padding=1)
        self.final_conv3 = nn.Conv2d(64, 1, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()



    def forward(self, x0):

        x0 = self.coars_enc_conv_s1(x0)
        x0 = self.relu(self.bn1(x0))


        # STAGE 2 1/2 1/2 
        x1 = self.maxpool(x0)
        x1 = conv_block(x1, self.conv_block2, self.relu)
        x1 = id_block(x1, self.id_block2_1, self.relu)
        # 1/2 1/2 256

        # STAGE 3 1/4 1/4
        x2 = self.maxpool(x1)
        x2 = conv_block(x2, self.conv_block3, self.relu)
        x2 = id_block(x2, self.id_block3_1, self.relu)
        x2 = id_block(x2, self.id_block3_2, self.relu)
        x2 = id_block(x2, self.id_block3_3, self.relu)
        # 1/4 1/4 512

        # STAGE 4 1/8 1/8
        x3 = self.maxpool(x2)
        x3 = conv_block(x3, self.conv_block4, self.relu)
        x3 = id_block(x3, self.id_block4_1, self.relu)
        x3 = id_block(x3, self.id_block4_2, self.relu)
        x3 = id_block(x3, self.id_block4_3, self.relu)
        x3 = id_block(x3, self.id_block4_4, self.relu)
        x3 = id_block(x3, self.id_block4_5, self.relu)
        # 1/8 1/8 1024

        # STAGE 5 1/16 1/16
        x4 = self.maxpool(x3)
        x4 = conv_block(x4, self.conv_block5, self.relu)
        x4 = id_block(x4, self.id_block5_1, self.relu)
        x4 = id_block(x4, self.id_block5_2, self.relu)
        # 1/16 1/16 2048


        #
        # START OF DECODER
        #

        x3 = decoder_block(x4, x3, self.decoder_block1, self.relu)
        x2 = decoder_block(x3, x2, self.decoder_block2, self.relu)
        x1 = decoder_block(x2, x1, self.decoder_block3, self.relu)
        x0 = decoder_block(x1, x0, self.decoder_block4, self.relu)



        #
        # START OF FINE NET
        #
        x4 = finenet_block(x4, self.fine_block1)
        x3 = finenet_block(x3, self.fine_block2)
        x2 = finenet_block(x2, self.fine_block3)
        x1 = finenet_block(x1, self.fine_block4)
        x0 = finenet_block(x0, self.fine_block5)

        x0 = self.final_conv1(torch.cat(
                [x4,x3,x2,x1,x0],
                1
            ))

        x0 = self.relu(self.final_bn1(x0))
        x0 = self.relu(self.final_conv2(x0))
        x0 = self.sigmoid(self.final_conv3(x0))
        return x0


from dataset import batch_generator
if __name__ == "__main__":
    m = IOGnet()
    g = batch_generator(1, 16, False, False)
    print(next(g))

    X, _, _ = next(g)
    print(X.shape)
    y_pred = m(X)
    print(y_pred.shape)
