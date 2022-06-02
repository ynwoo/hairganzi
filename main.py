import os, sys

from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout

import torch
import argparse
from SEAN.test import reconstruct
from face_parsing.test import parsing
from stargan_v2_master.core.wing import align_faces
from stargan_v2_master.test import image_create
from PIL import Image
import shutil

is_result = 0
gender = 'female'
def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form = resource_path('ui/proto1.ui') #메인 윈도우
form_class = uic.loadUiType(form)[0]

form_second = resource_path('ui/btn1_window.ui') #hairstyle
form_secondwindow = uic.loadUiType(form_second)[0]

form_third = resource_path('ui/btn2_window.ui') #dye   
form_thirdwindow = uic.loadUiType(form_third)[0]

form_fourth = resource_path('ui/btn3_window.ui') #preset
form_fourthwindow = uic.loadUiType(form_fourth)[0]

form_fifth = resource_path('ui/how_to_use_window.ui') #how to use
form_fifthwindow = uic.loadUiType(form_fifth)[0]

###

form_error1 = resource_path('ui/error1.ui') #error1
form_errorwindow1 = uic.loadUiType(form_error1)[0]

form_error3 = resource_path('ui/error3.ui') #error3
form_errorwindow3 = uic.loadUiType(form_error3)[0]


class WindowClass(QMainWindow, form_class): #메인 윈도우
    global is_result
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.radioButtonMale.clicked.connect(self.RadioButton)
        self.radioButtonFemale.clicked.connect(self.RadioButton)

    def btn_main_to_second(self):
        self.hide()                     # 메인윈도우 숨김
        self.second = secondwindow()    
        self.second.exec()              # 두번째 창을 닫을 때 까지 기다림
        self.show()                    # 두번째 창을 닫으면 다시 첫 번째 창이 보여짐
        self.show_main()
    
    def btn_main_to_third(self):
        self.hide()                    
        self.third = thirdwindow()    
        self.third.exec()              
        self.show()
        self.show_main()
        
    
    def btn_main_to_fourth(self):
        self.hide()                    
        self.fourth = fourthwindow()    
        self.fourth.exec()              
        self.show()
        self.show_main()
    
    def btn_how_to(self):
        self.fourth = fifthwindow()    
        self.fourth.exec()

    def RadioButton(self):
        global gender
        if self.radioButtonMale.isChecked():
            gender = 'male'
        
        elif self.radioButtonFemale.isChecked():
            gender = 'female'
	
    def show_main(self):
        if is_result == 1:
            qpixmap = QPixmap('image/result/synthesized_image/results_0.png')
            self.mainPhoto.setPixmap(qpixmap)
    
    #여기에 시그널-슬롯 연결 설정 및 함수 설정



class secondwindow(QDialog,QWidget,form_secondwindow): #hairstyle
    
    def __init__(self):
        args.mode = 'styling'
        super(secondwindow,self).__init__()
        self.initUi()
        self.show()

    def initUi(self):
        self.setupUi(self)
        self.isPic1 = 0
        self.isPic2 = 0
        self.hairstyleOriginalUpload.clicked.connect(self.fileopen1)
        self.hairstyleUpload.clicked.connect(self.fileopen2)
        self.hairstyleResult.clicked.connect(self.addimageRes)

    def fileopen1(self):
        global hairstyleOriginal_file
        hairstyleOriginal_file = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')
        
        if(hairstyleOriginal_file[0]):
            self.addimage1()
            shutil.copyfile(hairstyleOriginal_file[0], "image/ori/img/im/ori.jpg")
            shutil.copyfile(hairstyleOriginal_file[0], "image/ori/img/im2/ori.jpg")
            align_faces(args,"image/ori/img/im","image/ori/img/im")
            align_faces(args,"image/ori/img/im2","image/ori/img/im2")
            self.isPic1 = 1
        
    
    def fileopen2(self):
        global hairStyle_file
        hairStyle_file = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')
        print(type(hairStyle_file[0]))
        if(hairStyle_file[0]):
            self.addimage2()
            shutil.copyfile(hairStyle_file[0], "image/ref/img/im/ref.jpg")
            shutil.copyfile(hairStyle_file[0], "image/ref/img/im2/ref.jpg")
            align_faces(args,"image/ref/img/im","image/ref/img/im")
            align_faces(args,"image/ref/img/im2","image/ref/img/im2")
            self.isPic2 = 1

    def addimage1(self):
        qpixmap = QPixmap(hairstyleOriginal_file[0])
        self.originalLabel.setPixmap(qpixmap)
    
    def addimage2(self):
        qpixmap = QPixmap(hairStyle_file[0])
        self.originalLabel_2.setPixmap(qpixmap)

    def addimageRes(self):
        global is_result

        if(self.isPic1 and self.isPic2):
            image_create(args)
            image1 = Image.open('image/created_image/img/reference.jpg')

            if gender == 'female':
                croppedimgfemale = image1.crop((512,512,1024,1024))
                croppedimgfemale.save('image/created_image/img/cr_img.jpg')
            else:
                croppedimgmale = image1.crop((512,1024,1024,1536))
                croppedimgmale.save('image/created_image/img/cr_img.jpg')

            os.remove('image/created_image/img/reference.jpg')
            args.ref_respth = 'image/created_image/label'
            args.ref_depth = 'image/created_image/img'

            if(parsing(args.ref_respth,args.ref_depth,args.cp)):
            #ref image
                if(parsing(args.ori_respth,args.ori_depth,args.cp)):
                #org image
                    reconstruct(args.mode)
                else:
                    print("Wrong image select other pics")
            else:
                print("Wrong image select other pics")

            qpixmap = QPixmap('image/result/synthesized_image/results_0.png')
            self.hairstyleResultImg.setPixmap(qpixmap)
            is_result = 1
            
        else:
            self.error3 = errorwindow3() #원본 또는 스타일 이미지 없이 합성 누를때
    
    def btn_second_to_main(self):
        self.close()


    
class thirdwindow(QDialog,QWidget,form_thirdwindow): #dye
    def __init__(self):
        args.mode = 'dyeing'
        super(thirdwindow,self).__init__()
        self.initUi()
        self.show()

    def initUi(self):
        self.setupUi(self)
        self.isPic1 = 0
        self.isPic2 = 0
        self.dyeOriginalUpload.clicked.connect(self.fileopen1)
        self.dyeStyleUpload.clicked.connect(self.fileopen2)
        self.dyeResult.clicked.connect(self.addimageRes)
        

    def fileopen1(self):
        global dyeOriginal_file
        dyeOriginal_file = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')
        if(dyeOriginal_file[0]):
            self.addimage1()
            shutil.copyfile(dyeOriginal_file[0], "image/ori/img/im/ori.jpg")
            self.isPic1 = 1
        
    def fileopen2(self):
        global dyeStyle_file
        dyeStyle_file = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')
        if(dyeStyle_file[0]):
            self.addimage2()
            shutil.copyfile(dyeStyle_file[0], "image/ref/img/im/ref.jpg")
            self.isPic2 = 1

    def addimage1(self):
        qpixmap = QPixmap(dyeOriginal_file[0])
        self.originalLabel.setPixmap(qpixmap)
    
    def addimage2(self):
        qpixmap = QPixmap(dyeStyle_file[0])
        self.originalLabel_2.setPixmap(qpixmap)

    def addimageRes(self):
        global is_result
        if(self.isPic1 and self.isPic2):
            #ref image
            if(parsing(args.ref_respth,args.ref_depth,args.cp)):
                #org image
                if(parsing(args.ori_respth,args.ori_depth,args.cp)):
                    reconstruct(args.mode)
                else:
                    print("Wrong image select other pics")
            else:
                print("Wrong image select other pics")
            
            qpixmap = QPixmap('image/result/synthesized_image/results_0.png')
            self.dyeResultImg.setPixmap(qpixmap)
            is_result = 1
        else:
            self.error3 = errorwindow3() #원본 또는 스타일 이미지 없이 합성 누를때
    
    def btn_third_to_main(self):
        self.close()



class fourthwindow(QDialog,QWidget,form_fourthwindow): #preset
    global gender
    def __init__(self):
        args.mode = 'styling'
        super(fourthwindow,self).__init__()
        self.initUi()
        if gender == 'male': #남자일때 프리셋
            self.presetCombo.addItem('male1')
            self.presetCombo.addItem('male2')
            self.presetCombo.addItem('male3')
            self.curgender.setText('남자')
        else: #여자일때 프리셋
            self.presetCombo.addItem('female1')
            self.presetCombo.addItem('female2')
            self.presetCombo.addItem('female3')
            self.curgender.setText('여자')
        self.preset_value = str(self.presetCombo.currentText())  # 선택된 프리셋 값 가져오는 코드 = preset_value
        self.show()
        

    def initUi(self):
        self.setupUi(self)
        self.isPic = 0
        self.presetUpload.clicked.connect(self.fileopen)
        self.presetSyn.clicked.connect(self.addimageRes)
        

    def fileopen(self):
        global preset_file
        preset_file = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')
        if(preset_file[0]):
            self.addimage()
            shutil.copyfile(preset_file[0], "image/ori/img/im/ori.jpg")
            shutil.copyfile(preset_file[0], "image/ori/img/im2/ori.jpg")
            self.isPic = 1

    def addimage(self):
        qpixmap = QPixmap(preset_file[0])
        self.originalLabel.setPixmap(qpixmap)
        

    def addimageRes(self):
        global is_result

        if(self.isPic):
            # 어떤 프리셋인지에 따라 불러오는 이미지 다르게
            preset_ref_image = 'image/preset/' + str(self.presetCombo.currentText()) + '.jpg'

            shutil.copyfile(preset_ref_image, "image/ref/img/im/ref.jpg")
            shutil.copyfile(preset_ref_image, "image/ref/img/im2/ref.jpg")
            
            image_create(args)
            image1 = Image.open('image/created_image/img/reference.jpg')
 
            if gender == 'female':
                croppedimgfemale = image1.crop((512,512,1024,1024))
                croppedimgfemale.save('image/created_image/img/cr_img.jpg')
            else:
                croppedimgmale = image1.crop((512,1024,1024,1536))
                croppedimgmale.save('image/created_image/img/cr_img.jpg')

            os.remove('image/created_image/img/reference.jpg')
            args.ref_respth = 'image/created_image/label'
            args.ref_depth = 'image/created_image/img'

            #ref image
            if(parsing(args.ref_respth,args.ref_depth,args.cp)):
                #org image
                if(parsing(args.ori_respth,args.ori_depth,args.cp)):
                    reconstruct(args.mode)
                else:
                    print("Wrong image select other pics")
            else:
                print("Wrong image select other pics")

            qpixmap = QPixmap('image/result/synthesized_image/results_0.png')
            self.presetResultImg.setPixmap(qpixmap)
            is_result = 1
            
        else:
            self.error3 = errorwindow3()  # 원본 또는 스타일 이미지 없이 합성 누를때
    
    def btn_fourth_to_main(self):
        self.close()

class fifthwindow(QDialog,QWidget,form_fifthwindow):  # 사용법
    def __init__(self):
        super(fifthwindow,self).__init__()
        self.initUi()
        self.show()

    def initUi(self):
        self.setupUi(self)

###에러 창 모음
class errorwindow1(QDialog,QWidget,form_errorwindow1): #error1
    def __init__(self):
        super(errorwindow1,self).__init__()
        self.initUi()
        self.show()

    def initUi(self):
        self.setupUi(self)

    def okay(self):
        self.close()

class errorwindow3(QDialog,QWidget,form_errorwindow3): #error1
    def __init__(self):
        super(errorwindow3,self).__init__()
        self.initUi()
        self.show()

    def initUi(self):
        self.setupUi(self)
    
    def okay(self):
        self.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # implement
    parser.add_argument('--mode', type=str, default='styling',
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

    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
