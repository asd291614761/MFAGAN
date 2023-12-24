# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import logging
import os
import cv2
from Generator import Generator
from data import test_dataset
from options import opt
from torchvision.utils import save_image
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_G = Generator()

model_G.load_state_dict(torch.load(opt.model_G_Parameter))

print('Loading base network...')

model_G.cuda()

test_data_root = opt.test_data_root
maps_path = opt.maps_path

test_sets = ['VT5000/Test','VT1000','VT821']
mae = []
for dataset in test_sets:

    save_path = maps_path + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset_path = test_data_root + dataset
    test_loader = test_dataset(dataset_path, opt.size)
    mae_sum = 0
    for i in range(test_loader.size):
        image, t, gt, name= test_loader.load_data()
        image, t, gt = image.cuda(),t.cuda(),gt.cuda()
        gt = gt.squeeze()
        with torch.no_grad():
            rgb_feature, thermal_feature, feature_final = model_G(image,t)
        res = feature_final

        res = F.interpolate(res, size=gt.shape, mode='bilinear')

        res = res.sigmoid()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
        mae_sum += torch.sum(torch.abs(res - gt)) / torch.numel(gt)
        res = F.interpolate(res, size=(480,640),mode='bilinear')
        res = res.detach().cpu().numpy().squeeze()
        cv2.imwrite(save_path + name,res*255)

        print("test-dataset:{}, num:{} ".format(dataset,i))
    mae1 = mae_sum / test_loader.size

    mae.append(mae1)
logging.basicConfig(filename=opt.maps_path + 'test-log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info('VT5000:{};VT1000:{};VT821:{}'.format(mae[0],mae[1],mae[2]))

print(mae)