"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

from SEAN import data
from SEAN.options.test_options import TestOptions
from SEAN.models.pix2pix_model import Pix2PixModel
from SEAN.util.visualizer import Visualizer
from itertools import cycle

def reconstruct(mode):
    opt = TestOptions().parse()
    opt.status = 'test'
    
    opt.contain_dontcare_label = True
    opt.no_instance = True
    
    src_dataloader = data.create_dataloader(opt)

    model = Pix2PixModel(opt)
    model.eval()
    visualizer = Visualizer(opt)

    if mode == 'dyeing':
        opt.styling_mode = mode
        opt.image_dir = './image/ref/img/im'
        opt.label_dir = './image/ref/label'

    else:
        opt.styling_mode = 'styling'
        opt.image_dir = './image/created_image/img'
        opt.label_dir = './image/created_image/label'

    res_dataloader = data.create_dataloader(opt)

    for i, data_i in enumerate(zip(cycle(src_dataloader), res_dataloader)):
        src_data = data_i[0]
        ref_data = data_i[1]
        generated = model(src_data, ref_data, mode=opt.styling_mode)

        img_path = src_data['path']
        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([('input_label', src_data['label'][b]),
                               ('synthesized_image', generated[b])])

            visualizer.save_images(visuals, opt.results_dir, f'results_{i}')
    