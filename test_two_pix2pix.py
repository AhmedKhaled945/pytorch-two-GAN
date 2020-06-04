import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
#from util.visualizer import Visualizer
#from util import html
import random
import torchvision.transforms as transforms
import cv2
from PIL import Image


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.continue_train = False


AB_path = '/content/pytorch-two-GAN/datasets/soccer_seg_detection/train_phase_1/001_AB.jpg'

AB = Image.open(AB_path).convert('RGB')
AB = AB.resize((256 * 2, 256), Image.BICUBIC)
AB = transforms.ToTensor()(AB)

w_total = AB.size(2)
w = int(w_total / 2)
h = AB.size(1)
w_offset = random.randint(0, max(0, w - 256 - 1))
h_offset = random.randint(0, max(0, h - 256 - 1))

A = AB[:, h_offset:h_offset + 256,
        w_offset:w_offset + 256]
B = AB[:, h_offset:h_offset + 256,
        w + w_offset:w + w_offset + 256]

A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

input_nc = 3
output_nc = 3

#if (not self.opt.no_flip) and random.random() < 0.5:
#    idx = [i for i in range(A.size(2) - 1, -1, -1)]
#    idx = torch.LongTensor(idx)
#    A = A.index_select(2, idx)
#    B = B.index_select(2, idx)

if input_nc == 1:  # RGB to gray
    tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
    A = tmp.unsqueeze(0)

if output_nc == 1:  # RGB to gray
    tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
    B = tmp.unsqueeze(0)

data = {'A': A.unsqueeze(0), 'B': B.unsqueeze(0),
        'A_paths': AB_path, 'B_paths': AB_path}
        


model = create_model(opt)


model.set_input(data)    
model.test()

visuals = model.get_current_visuals()

dst_dir = '/content/output/'

cv2.imwrite(dst_dir+'real_A.png',visuals['real_A'])
cv2.imwrite(dst_dir+'real_D.png',visuals['real_D'])
cv2.imwrite(dst_dir+'fake_B.png',visuals['fake_B'])
cv2.imwrite(dst_dir+'fake_C.png',visuals['fake_C'])
cv2.imwrite(dst_dir+'fake_D.png',visuals['fake_D'])
