import os
import torch
import torch.nn.functional as F
import numpy as np
from Generator import Generator
from Discriminator import Discriminator
from data import train_dataloader,val_dataloader
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
from loss.ssim import ssim
from _datetime import datetime
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.benchmark = True
cudnn.enabled = True
seed = torch.initial_seed()
def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()
def bound_iou_loss(pred, mask):
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()

def tesnor_bound(img, ksize):
    B, C, H, W = img.shape
    pad = int((ksize - 1) // 2)
    img_pad = F.pad(img, pad=[pad, pad, pad, pad], mode='constant',value = 0)
    patches = img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    corrosion, _ = torch.min(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    inflation, _ = torch.max(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    return inflation - corrosion

def train(train_loader, model_G,model_D, optimizer_G,optimizer_D, epoch, save_path):

    model_G.train()
    model_D.train()
    total_step = len(train_loader)
    for i, (images, thermals, gts, bodys, details) in enumerate(train_loader):
        batch_num = images.size()[0]

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        image, thermal, gt, body, detail = images.cuda(), thermals.cuda(), gts.cuda(), bodys.cuda(), details.cuda()

        real_labels = torch.ones(size=[batch_num,1], dtype=torch.float32, device=gt.device)
        fake_labels = torch.zeros(size=[batch_num,1], dtype=torch.float32, device=gt.device)

        rgb_feature, thermal_feature, feature_final = model_G(image, thermal)

        pre_dic = model_D(feature_final)

        gt0 = F.interpolate(gt, (14, 14), mode="bilinear")
        gt1 = F.interpolate(gt, (28, 28), mode="bilinear")
        gt2 = F.interpolate(gt, (56, 56), mode="bilinear")
        gt3 = F.interpolate(gt, (112, 112), mode="bilinear")

        gt_bound  = tesnor_bound(gt,3).cuda()
        gt_bound3 = tesnor_bound(gt3,3).cuda()
        gt_bound2 = tesnor_bound(gt2,3).cuda()
        gt_bound1 = tesnor_bound(gt1,3).cuda()
        gt_bound0 = tesnor_bound(gt0,3).cuda()

        f_final_bound   = tesnor_bound(torch.sigmoid(feature_final),      3).cuda()
        t_bound3        = tesnor_bound(torch.sigmoid(thermal_feature[3]), 3).cuda()
        t_bound2        = tesnor_bound(torch.sigmoid(thermal_feature[2]), 3).cuda()
        t_bound1        = tesnor_bound(torch.sigmoid(thermal_feature[1]), 3).cuda()
        t_bound0        = tesnor_bound(torch.sigmoid(thermal_feature[0]), 3).cuda()

        r_bound3        = tesnor_bound(torch.sigmoid(rgb_feature[3]),     3).cuda()
        r_bound2        = tesnor_bound(torch.sigmoid(rgb_feature[2]),     3).cuda()
        r_bound1        = tesnor_bound(torch.sigmoid(rgb_feature[1]),     3).cuda()
        r_bound0        = tesnor_bound(torch.sigmoid(rgb_feature[0]),     3).cuda()

        rgb_loss1 = F.binary_cross_entropy_with_logits(rgb_feature[0], gt0) + iou_loss(rgb_feature[0],gt0)
        rgb_loss2 = F.binary_cross_entropy_with_logits(rgb_feature[1], gt1) + iou_loss(rgb_feature[1],gt1)
        rgb_loss3 = F.binary_cross_entropy_with_logits(rgb_feature[2], gt2) + iou_loss(rgb_feature[2],gt2)
        rgb_loss4 = F.binary_cross_entropy_with_logits(rgb_feature[3], gt3) + iou_loss(rgb_feature[3],gt3)
        t_loss1   = F.binary_cross_entropy_with_logits(thermal_feature[0], gt0) + iou_loss(thermal_feature[0],gt0)
        t_loss2   = F.binary_cross_entropy_with_logits(thermal_feature[1], gt1) + iou_loss(thermal_feature[1],gt1)
        t_loss3   = F.binary_cross_entropy_with_logits(thermal_feature[2], gt2) + iou_loss(thermal_feature[2],gt2)
        t_loss4   = F.binary_cross_entropy_with_logits(thermal_feature[3], gt3) + iou_loss(thermal_feature[3],gt3)

        bound_loss = bound_iou_loss(f_final_bound, gt_bound) + \
                     bound_iou_loss(r_bound3, gt_bound3) + \
                     bound_iou_loss(r_bound2, gt_bound2) + \
                     bound_iou_loss(r_bound1, gt_bound1) + \
                     bound_iou_loss(r_bound0, gt_bound0) + \
                     bound_iou_loss(t_bound3, gt_bound3) + \
                     bound_iou_loss(t_bound2, gt_bound2) + \
                     bound_iou_loss(t_bound1, gt_bound1) + \
                     bound_iou_loss(t_bound0, gt_bound0)

        f_gt_loss = F.binary_cross_entropy_with_logits(feature_final, gt) + iou_loss(feature_final,gt)

        adv_loss_funtion = torch.nn.BCELoss()
        G_dis_loss = adv_loss_funtion(pre_dic, real_labels)

        loss_all_G = rgb_loss1 + rgb_loss2 + rgb_loss3 + rgb_loss4 + t_loss1 + t_loss2 + t_loss3 + t_loss4 + f_gt_loss + bound_loss + G_dis_loss

        if i % 50 == 0 or i == total_step or i==1:
            print("{}, G :Epoch [{}/{}], Step [{}/{}], R_loss_all:{:.4f}, t_loss_all:{:.4f}, f_gt_loss:{:.4f}, bound_loss:{:.4f}, G_dis_loss:{:.4f}".
                  format(datetime.now().strftime('%H:%M:%S') ,epoch, opt.epoch, i, total_step, rgb_loss1+rgb_loss2+rgb_loss3+rgb_loss4, t_loss1+t_loss2+t_loss3+t_loss4, f_gt_loss, bound_loss,G_dis_loss ))
            logging.info('Epoch [{}/{}], Step [{}/{}], R_loss_all:{:.4f}, t_loss_all:{:.4f}, f_gt_loss:{:.4f}, bound_loss:{:.4f}, G_dis_loss:{:.4f}'.
                         format(epoch, opt.epoch, i, total_step, rgb_loss1+rgb_loss2+rgb_loss3+rgb_loss4, t_loss1+t_loss2+t_loss3+t_loss4, f_gt_loss, bound_loss, G_dis_loss))

        loss_all_G.backward()
        optimizer_G.step()

        pre_dic_d = model_D(feature_final.detach())
        real_dic = model_D(gt)
        pre_loss = adv_loss_funtion(pre_dic_d, fake_labels)
        real_loss = adv_loss_funtion(real_dic, real_labels)
        loss_all_D = pre_loss + real_loss
        if i % 50 == 0 or i == total_step or i==1:
            print("{}, D :Epoch [{}/{}], Step [{}/{}], pre_loss:{:.4f}, real_loss:{:.4f}, loss_all:{:.4f}".
                  format(datetime.now().strftime('%H:%M:%S') ,epoch, opt.epoch, i, total_step,pre_loss, real_loss,loss_all_D))
            logging.info("D :Epoch [{}/{}], Step [{}/{}], pre_loss:{:.4f}, real_loss:{:.4f}, loss_all:{:.4f}".
                         format(epoch, opt.epoch, i, total_step,pre_loss, real_loss,loss_all_D))
        loss_all_D.backward()
        optimizer_D.step()

def val(val_loader, model_G, save_path, epoch):
    global best_mae,best_epoch
    model_G.eval()
    with torch.no_grad():
        mae_sum=0
        for i, (images, thermals, gts, name, _) in enumerate(val_loader):
            image, t, gt = images.cuda(), thermals.cuda(), gts.cuda()
            rgb_feature, thermal_feature, feature_final = model_G(image,t)
            res = feature_final
            res = F.interpolate(res, size=(gt.shape[2], gt.shape[3]), mode='bilinear')
            res = res.sigmoid()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += torch.sum(torch.abs(res - gt)) / torch.numel(gt)

        mae = mae_sum/len(val_loader)
        if epoch==0:
            best_mae=mae
        else:
            if mae < best_mae:
                best_mae=mae
                best_epoch=epoch
                torch.save(model_G.state_dict(), save_path+'Generator_epoch_{}_best.pth'.format(epoch))
        print("-------------------------------val--------------------------------")
        print("Epoch:{}, MAE:{:.5f}, bestEpoch:{}, BestMAE:{:.5f}".format(epoch, mae, best_epoch, best_mae))
        logging.info('##Val##:Epoch:{}   MAE:{}   bestEpoch:{}   bestMAE:{}'.format(epoch,mae,best_epoch,best_mae))

if __name__ == '__main__':
    model_G = Generator()
    model_D = Discriminator()
    if opt.load is not None:
        model_G.load_pre(opt.load)
        print('load model from ', opt.load)
    model_G = model_G.cuda()
    model_D = model_D.cuda()
    params_G = model_G.parameters()
    optimizer_G = torch.optim.Adam(params_G, opt.lr_G)
    params_D = model_D.parameters()
    optimizer_D = torch.optim.Adam(params_D, opt.lr_D)

    train_root = opt.train_data_root
    val_root = opt.val_data_root

    log_save_path = opt.result_save_path
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
    Parameter_save_path = opt.Parameter_save_path
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)

    train_loader = train_dataloader(train_root, batchsize=opt.train_batchsize, size=opt.size)
    val_loader   = val_dataloader(val_root,batchsize = opt.test_batchsize, size=opt.size)

    best_mae = 0
    best_epoch = 0
    logging.basicConfig(filename=log_save_path + 'train_log_v4.0', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("----------Begin---------")
    logging.info('epoch:{};lr:{};batchsize:{};size:{};clip:{};seed:{}'.format(opt.epoch, opt.lr_G, opt.train_batchsize, opt.size, opt.clip, seed))
    writer = SummaryWriter(log_save_path + 'summary')

    for epoch in range(opt.epoch):
        train(train_loader, model_G,model_D,  optimizer_G,optimizer_D, epoch, Parameter_save_path)
        val(val_loader,model_G,Parameter_save_path,epoch)


