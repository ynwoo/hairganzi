"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from SEAN.models.networks.base_network import BaseNetwork
from SEAN.models.networks.normalization import get_nonspade_norm_layer
from SEAN.models.networks.architecture import ResnetBlock as ResnetBlock
from SEAN.models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from SEAN.models.networks.architecture import Zencoder

class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        self.Zencoder = Zencoder(3, 512)


        self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt, Block_Name='head_0')

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt, Block_Name='G_middle_0')
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt, Block_Name='G_middle_1')

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt, Block_Name='up_0')
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt, Block_Name='up_1')
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt, Block_Name='up_2')
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt, Block_Name='up_3', use_rgb=False)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt, Block_Name='up_4')
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear')


    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def dyeing(self,og_seg,og_img,nog_seg,nog_img,obj_dic):
        x = F.interpolate(og_seg,size =(self.sh, self.sw))
        x = self.fc(x)
        
        style_codes = self.Zencoder(input=og_img, segmap=og_seg)
        dyeing_style_codes = self.Zencoder(input=nog_img, segmap=nog_seg)

        style_codes[0][13] = dyeing_style_codes[0][13]

        
        x = self.head_0(x, og_seg, style_codes, obj_dic=obj_dic)

        x = self.up(x)
        x = self.G_middle_0(x, og_seg, style_codes, obj_dic=obj_dic)
         
        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, og_seg, style_codes,  obj_dic=obj_dic)

        x = self.up(x)
        x = self.up_0(x, og_seg, style_codes, obj_dic=obj_dic)
        x = self.up(x)
        x = self.up_1(x, og_seg, style_codes, obj_dic=obj_dic)
        x = self.up(x)
        x = self.up_2(x, og_seg, style_codes, obj_dic=obj_dic)
        x = self.up(x)
        x = self.up_3(x, og_seg, style_codes,  obj_dic=obj_dic)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x

    def styling(self,og_seg,og_img,nog_seg,obj_dic):
        x = F.interpolate(nog_seg,size =(self.sh, self.sw))
        x = self.fc(x)

        style_codes = self.Zencoder(input=og_img, segmap=og_seg)


        x = self.head_0(x, nog_seg, style_codes, obj_dic=obj_dic)

        x = self.up(x)
        x = self.G_middle_0(x, nog_seg, style_codes, obj_dic=obj_dic)
         
        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, nog_seg, style_codes,  obj_dic=obj_dic)

        x = self.up(x)
        x = self.up_0(x, nog_seg, style_codes, obj_dic=obj_dic)
        x = self.up(x)
        x = self.up_1(x, nog_seg, style_codes, obj_dic=obj_dic)
        x = self.up(x)
        x = self.up_2(x, nog_seg, style_codes, obj_dic=obj_dic)
        x = self.up(x)
        x = self.up_3(x, nog_seg, style_codes,  obj_dic=obj_dic)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x
