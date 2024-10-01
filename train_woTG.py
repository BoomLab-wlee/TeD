import os
import random
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import skimage.io as skio

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.data_loader import gen_train_dataloader
from utils.util import parse_arguments, imshow, grad_imshow
from model.TeD_woTG import TeD

from utils.sampling import generate_mask_pair
from utils.sampling import generate_subimages
from utils.loss_function import VGGLoss

def train(train_dataloader, model, optimizer, writer, epoch, opt, device):
    """
    Train a model for a single epoch

    Arguments:
        train_dataloader: (Pytorch DataLoader)
        model: (Pytorch nn.Module)
        optimizer: (Pytorch optimzer)
        rng: numpy random number generator
        writer: (Tensorboard writer)
        epoch: epoch of training (int)
        opt: argparse dictionary

    Returns:
        loss_list: list of total loss of each batch ([float])
        loss_list_l1: list of L1 loss of each batch ([float])
        loss_list_l2: list of L2 loss of each batch ([float])
        corr_list: list of correlation of each batch ([float])
    """
    # initialize
    model.train()
    loss_list_l1 = []
    loss_list_l2 = []
    loss_list_reg = []
    loss_list = []

    L1_pixelwise = torch.nn.L1Loss()
    L2_pixelwise = torch.nn.MSELoss()

    loss_coef = opt.loss_coef

    # training
    for i, data in enumerate(tqdm(train_dataloader)):

        (noisy_image, tempGradMap, _, mean, std, name) = data

        B, T, X, Y = noisy_image.shape
        noisy_image, tempGradMap = noisy_image.to(device), tempGradMap.to(device)
        target_image = noisy_image[:, int(T / 2), :].unsqueeze(dim=1)

        mask1, mask2 = generate_mask_pair(noisy_image)

        sub_noisy_image = generate_subimages(noisy_image, mask1)
        sub_tempGradMap = generate_subimages(tempGradMap, mask1)

        sub_target_image =  generate_subimages(target_image, mask2)

        optimizer.zero_grad()

        sub_denoised_image = model(sub_noisy_image, sub_tempGradMap)
        loss_l1_pixelwise = L1_pixelwise(sub_denoised_image, sub_target_image)
        loss_l2_pixelwise = L2_pixelwise(sub_denoised_image, sub_target_image)

        # Regularizer
        with torch.no_grad():
            denoised_image = model(noisy_image, tempGradMap)

        sub_denoised_image1 = generate_subimages(denoised_image, mask1)
        sub_denoised_image2 = generate_subimages(denoised_image, mask2)

        reg_loss_l1_pixelwise = L1_pixelwise(sub_denoised_image-sub_target_image
                                             , sub_denoised_image1-sub_denoised_image2)
        reg_loss_l2_pixelwise = L2_pixelwise(sub_denoised_image-sub_target_image
                                             , sub_denoised_image1-sub_denoised_image2)

        loss_reg = loss_coef[0] * reg_loss_l1_pixelwise + loss_coef[1] * reg_loss_l2_pixelwise
        Lambda = opt.reg_loss_ratio
        loss_sum = loss_coef[0] * loss_l1_pixelwise + loss_coef[1] * loss_l2_pixelwise + Lambda * loss_reg

        loss_sum.backward()
        optimizer.step()

        loss_list_l1.append(loss_l1_pixelwise.item())
        loss_list_l2.append(loss_l2_pixelwise.item())

        loss_list_reg.append(loss_reg.item())
        loss_list.append(loss_sum.item())

        # print log
        if (epoch % opt.logging_interval == 0) and (i % opt.logging_interval_batch == 0):
            loss_mean = np.mean(np.array(loss_list))
            loss_mean_l1 = np.mean(np.array(loss_list_l1))
            loss_mean_l2 = np.mean(np.array(loss_list_l2))
            loss_mean_reg = np.mean(np.array(loss_list_reg))

            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            writer.add_scalar("Loss_l1/train_batch", loss_mean_l1, epoch * len(train_dataloader) + i)
            writer.add_scalar("Loss_l2/train_batch", loss_mean_l2, epoch * len(train_dataloader) + i)
            writer.add_scalar("Loss_reg/train_batch", loss_mean_reg, epoch * len(train_dataloader) + i)
            writer.add_scalar("Loss/train_batch", loss_mean, epoch * len(train_dataloader) + i)

            logging.info(f"[{ts}] Epoch [{epoch}/{opt.n_epochs}] Batch [{i + 1}/{len(train_dataloader)}] " + \
                         f"loss : {loss_mean:.4f}, loss_l1 : {loss_mean_l1:.4f}, loss_l2 : {loss_mean_l2:.4f} " +
                         f"loss_reg : {loss_mean_reg:.4f}"
                         )

    if epoch % 10 == 0 or epoch == 0:
        f_idx = random.randint(0, B - 1)
        result_show = torch.cat((noisy_image[f_idx,int(T / 2),:,:].squeeze()*std[f_idx].squeeze()+mean[f_idx].squeeze(),
                       denoised_image[f_idx,:].squeeze()*std[f_idx].squeeze() + mean[f_idx].squeeze()), dim=1)
        imshow(result_show.detach().cpu(), title='[%s] EPOCH : %s' % (opt.save_name, epoch))

    return loss_list, loss_list_l1, loss_list_l2, loss_list_reg


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    # ----------
    # Initialize: Create sample and checkpoint directories
    # ----------
    opt = parse_arguments()
    cuda = torch.cuda.is_available() and (not opt.use_CPU)
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    rng = np.random.default_rng(opt.random_seed)

    os.makedirs(opt.results_dir + "/images/{}".format(opt.save_name), exist_ok=True)
    os.makedirs(opt.results_dir + "/saved_models/{}".format(opt.save_name), exist_ok=True)
    os.makedirs(opt.results_dir + "/logs".format(opt.save_name), exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=opt.results_dir + "/logs/{}.log".format(opt.save_name), \
                        filemode="a", format="%(name)s - %(levelname)s - %(message)s")
    writer = SummaryWriter(opt.results_dir + "/tsboard/{}".format(opt.save_name))

    # ----------
    # Dataset
    # ----------
    dataloader_train = gen_train_dataloader(opt.root, opt.image_size, True, opt.batch_num, True)

    (noisy_image, tempGradMap, _, mean, std, name) = next(iter(dataloader_train))

    B, T, X, Y = noisy_image.shape
    f_idx = random.randint(0, opt.batch_num - 1)
    imshow(noisy_image[f_idx,int(T / 2),:].squeeze()*std[f_idx].squeeze() + mean[f_idx].squeeze())
    grad_imshow(tempGradMap[f_idx,int(T / 2),:].squeeze())

    # ----------
    # Model, Optimizers, and Loss
    # ----------
    # 240426 model
    # model = SwinIR(img_size=(opt.patch_size[1]//2, opt.patch_size[2]//2),
    #                in_channels=opt.input_frames,
    #                out_channels=1,
    #                window_size=15, depths=[6, 6, 6, 6, 6],
    #                embed_dim=128, num_heads=8, mlp_ratio=4,
    #                attn_drop_rate=0.1)

    # light weight model / 240808 model

    model = TeD(img_size=(opt.image_size[1]//2, opt.image_size[2]//2),
                   patch_size=opt.patch_size,
                   in_channels=opt.input_frames,
                   out_channels=opt.output_frames,
                   window_size=opt.window_size, depths=opt.rstb_depths,
                   embed_dim=opt.embed_dim,
                   num_heads=opt.num_heads,
                   attn_drop_rate=opt.attn_drop_rate)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    if cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = nn.DataParallel(model, device_ids=[0], output_device=0)
        model.to(device)


    if opt.epoch != 0:
        if opt.epoch <= opt.n_epochs // 5:
            load_epoch = 0
        else:
            load_epoch = opt.epoch - 1
        model.load_state_dict(
            torch.load(opt.results_dir + "/saved_models/%s/model_%d.pth" % (opt.save_name, load_epoch)))
        optimizer.load_state_dict(
            torch.load(opt.results_dir + "/saved_models/%s/optimizer_%d.pth" % (opt.save_name, load_epoch)))
        print('Loaded pre-trained model and optimizer weights of epoch {}'.format(opt.epoch - 1))


    # ----------
    # Training & Validation
    # ----------
    for epoch in range(opt.epoch, opt.n_epochs):
        tic = time.time()
        loss_list, loss_list_l1, loss_list_l2, loss_list_reg = \
            train(dataloader_train, model, optimizer, writer, epoch, opt, device)

        # logging
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        if (epoch % opt.logging_interval == 0):
            loss_mean = np.mean(np.array(loss_list))
            loss_mean_l1 = np.mean(np.array(loss_list_l1))
            loss_mean_l2 = np.mean(np.array(loss_list_l2))
            loss_mean_reg = np.mean(np.array(loss_list_reg))

            writer.add_scalar("Loss/train", loss_mean, epoch)
            writer.add_scalar("Loss_l1/train", loss_mean_l1, epoch)
            writer.add_scalar("Loss_l2/train", loss_mean_l2, epoch)
            writer.add_scalar("Loss_reg/train", loss_mean_reg, epoch)

            logging.info(f"[{ts}] Epoch [{epoch}/{opt.n_epochs}] " + \
                         f"loss : {loss_mean:.4f}, loss_l1 : {loss_mean_l1:.4f}, loss_l2 : {loss_mean_l2:.4f},"
                         f"loss_reg : {loss_mean_reg:.4f}")

            print(f"[{ts}] Epoch [{epoch}/{opt.n_epochs}] " + \
                         f"loss : {loss_mean:.4f}, loss_l1 : {loss_mean_l1:.4f}, loss_l2 : {loss_mean_l2:.4f},"
                         f"loss_reg : {loss_mean_reg:.4f}, Time : {time.time() - tic:.4f} "
                  )

        if (opt.checkpoint_interval != -1) and (epoch % opt.checkpoint_interval == 0):
            torch.save(model.state_dict(), opt.results_dir + "/saved_models/%s/model_%d.pth" % (opt.save_name, epoch))
            torch.save(optimizer.state_dict(),
                       opt.results_dir + "/saved_models/%s/optimizer_%d.pth" % (opt.save_name, epoch))