#!/usr/bin/python
# -*- encoding: utf-8 -*-

from pickle import TRUE
from face_parsing.model import BiSeNet
import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
    


def parsing(respth='./res/test_res', dspth='./data', cp='79999_iter.pth'): #저장위치, 이미지 불러오는 위

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    #net.cuda()
    save_pth = osp.join('face_parsing/res/cp', cp)
    net.load_state_dict(torch.load(save_pth,map_location=torch.device('cpu')))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    print(dspth)
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            
            li = np.unique(parsing).tolist()

            print(np.unique(parsing))
            parsing = parsing + 100
            parsing[parsing == 100] = 0
            parsing[parsing == 101] = 1
            parsing[parsing == 110] = 2
            parsing[parsing == 106] = 3
            parsing[parsing == 104] = 4
            parsing[parsing == 105] = 5
            parsing[parsing == 102] = 6
            parsing[parsing == 103] = 7
            parsing[parsing == 107] = 8
            parsing[parsing == 108] = 9
            parsing[parsing == 111] = 10
            parsing[parsing == 112] = 11
            parsing[parsing == 113] = 12
            parsing[parsing == 117] = 13
            parsing[parsing == 118] = 14
            parsing[parsing == 109] = 15
            parsing[parsing == 115] = 16
            parsing[parsing == 114] = 17
            parsing[parsing == 116] = 18
            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))
            print(osp.join(respth, image_path))
            return TRUE
