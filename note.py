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
from utils.data_loader import gen_train_dataloader, random_transform
from utils.util import parse_arguments
from utils.util import imshow, grad_imshow
from utils.dataloader_test import BinomDataset

if __name__ == "__main__":
    # ----------
    # Initialize: Create sample and checkpoint directories
    # ----------
    opt = parse_arguments()

    # -----------
    # Dataset
    # ----------
    root = '/data/leewj/[2023-2024]Fluorescence_denoising/testset/240816_test'
    dataloader_train = gen_train_dataloader(root, [31, 400, 400], True, 1, True)
    (noisy_image, grad_map, _, mean, std, name) = next(iter(dataloader_train))
    minpsnr = -40
    maxpsnr = -5

    noisy_image = noisy_image*std+mean
    imshow(noisy_image[0,0,:].squeeze())
    noisy_image = np.clip(noisy_image.squeeze().detach().numpy(), 0, 255).astype('uint8')

    testloader = BinomDataset(noisy_image, 400, minpsnr, maxpsnr)
    test = next(iter(testloader))
    imshow(test[0,:].squeeze()*2.0)
    imshow(test[1,:].squeeze()*2.0)
    imshow(test[1,:].squeeze() + test[0,:].squeeze())
    new_test = np.abs(test[0,:].squeeze() - test[1,:].squeeze())
    print(new_test)
    imshow(new_test*20)

    # test_img = noisy_image[0,0,:].unsqueeze(dim=0)
    # img = test_img
    # uniform = np.random.rand() * (maxpsnr - minpsnr) + minpsnr
    #
    # level = (10 ** (uniform / 10.0)) / (img.type(torch.float).mean().item() + 1e-5)
    # level = min(level, 0.99)
    # print(level)
    # print(img.shape)
    # binom = torch.distributions.binomial.Binomial(total_count=img, probs=torch.tensor([level]))
    # imgNoise = binom.sample()
    #
    # img = (img - imgNoise)[None, ...].type(torch.float)
    # img = img / (img.mean() + 1e-8)
    #
    # imgNoise = imgNoise[None, ...].type(torch.float)
    #
    #
    # imshow(test_img)
    # imshow(img)
    # imshow(imgNoise)

