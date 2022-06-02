import torch
import argparse
from SEAN.test import reconstruct
from face_parsing.test import parsing
from stargan_v2_master.test import image_create
from stargan_v2_master.core.wing import align_faces
from PIL import Image
import os

def main_demo(args):
    torch.manual_seed(args.seed)
    if args.mode == 'styling':
        flag = 0  # 0: female, 1: male

        image_create(args)
        image1 = Image.open('image/created_image/img/reference.jpg')

        if flag == 0:
            croppedimgfemale = image1.crop((512,512,1024,1024))
            croppedimgfemale.save('image/created_image/img/cr_img.jpg')
        else:
            croppedimgmale = image1.crop((512,1024,1024,1536))
            croppedimgmale.save('image/created_image/img/cr_img.jpg')
        
        os.remove('image/created_image/img/reference.jpg')
        args.ref_respth = 'image/created_image/label'
        args.ref_depth = 'image/created_image/img'
        align_faces(args,args.ref_respth,args.ref_respth)
        align_faces(args,args.ori_respth,args.ori_respth)
        #ref image
        if(parsing(args.ref_respth,args.ref_depth,args.cp)):
            #org image
            if(parsing(args.ori_respth,args.ori_depth,args.cp)):
                reconstruct(args.mode)
            else:
                print("Wrong image select other pics")
        else:
            print("Wrong image select other pics")

    elif args.mode == 'dyeing':
        #ref image
        if(parsing(args.ref_respth,args.ref_depth,args.cp)):
            #org image
            if(parsing(args.ori_respth,args.ori_depth,args.cp)):
                reconstruct(args.mode)
            else:
                print("Wrong image select other pics")
        else:
            print("Wrong image select other pics")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # implement
    parser.add_argument('--mode', type=str, required=True,
                        choices=['dyeing','styling'], help='Select mode')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # StarGAN_v2
    parser.add_argument('--img_size', type=int, default=512, help='Image resolution')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers used in DataLoader')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size for validation')
    parser.add_argument('--num_domains', type=int, default=2, help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,help='Style code dimension')
    parser.add_argument('--w_hpf', type=float, default=1, help='weight for high-pass filtering')

    parser.add_argument('--resume_iter', type=int, default=100000,help='number of iteration')
    parser.add_argument('--checkpoint_dir', type=str, default='pretrained_network/StarGAN')
    parser.add_argument('--wing_path', type=str, default='pretrained_network/StarGAN/wing.ckpt')

    parser.add_argument('--src_dir', type=str, default='./image/ori/img')
    parser.add_argument('--result_dir', type=str, default='./image/created_image/img')
    parser.add_argument('--lm_path', type=str, default='pretrained_network/StarGAN/celeba_lm_mean.npz')
    # hair styling
    parser.add_argument('--ref_dir', type=str, default='./image/ref/img')

    #face parsing
    parser.add_argument('--ori_respth',type=str, default='./image/ori/label',help = 'Original image location')
    parser.add_argument('--ori_depth',type = str,default ='./image/ori/img/im',help = 'original image label location')

    parser.add_argument('--ref_respth',type=str, default='./image/ref/label',help = 'ref image location')
    parser.add_argument('--ref_depth',type = str,default = './image/ref/img/im',help = 'ref image label location')
    parser.add_argument('--cp',type=str,default = '79999_iter.pth',help = 'face parsing pretrained model location')

    args = parser.parse_args()
    print(type(args.seed))
    main_demo(args)
