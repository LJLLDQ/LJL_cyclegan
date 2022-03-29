
import os
from models.cycle_gan_model import CycleGANModel
from easydict import EasyDict as edict
import torch
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
# from options.test_options import TestOptions
class NewModel(CycleGANModel):


    def set_input(self, input):
        self.real_A = input.to(self.device)


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        # self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        # self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        # self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))


        

def get_transform():

    osize = [286, 286]

    transform_list = []
    transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    transform_list.append(transforms.RandomCrop(256))
    transform_list += [transforms.ToTensor()]       
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] 

    return transforms.Compose(transform_list)


# self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

# hard-code some parameters for test
def get_opt():
    opt = edict()
    opt.isTrain = False
    opt.checkpoints_dir = '/home/jinliang/pytorch-CycleGAN-and-pix2pix/checkpoints/'
    opt.name = 'cyclegan_v1'
    opt.preprocess = None
    # opt.save_path = 
    opt.epoch = 5
    opt.load_iter = 0

    opt.input_nc = 3
    opt.output_nc = 3
    opt.ngf = 64
    opt.netG = 'resnet_9blocks'
    opt.norm = 'instance'
    opt.no_dropout = 'store_true'
    opt.init_type = 'normal'
    opt.init_gain = 0.02
    opt.gpu_ids = [0]

    return opt

def f(x):

    opt = get_opt()
    model = NewModel(opt)
    model.setup(opt)  

    model.eval()
    x = model.set_input(x)
    with torch.no_grad():
        model.forward()

    return model.fake_B


# x = torch.rand(1, 3,512,512).cuda()

x_path = '/home/jinliang/pytorch-CycleGAN-and-pix2pix/cyclegan_train_data/trainA/5.jpg'

x = Image.open(x_path).convert('RGB')
# x = 
# x = torch.from_numpy(x)
x = get_transform()(x).unsqueeze(0)
result = f(x)


result = result[0].cpu().float().numpy()  # convert it into a numpy array
result = (np.transpose(result, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
result.astype(np.uint8)

result = cv2.imwrite('./1.jpg', result)
# image_numpy.astype(np.uint8)



# print(result.shape)



