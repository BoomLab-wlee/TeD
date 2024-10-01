import skimage.io as skio
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from utils.util import get_coordinate, mean_filter_2d

class DataFolder(Dataset):
    def __init__(self, root, patch_size=[61, 128, 128], random_transform=True, random_patch_seed=0):
        """
        Arguments:
            noisy_images: list of noisy image stack ([Tensor with dimension [t, x, y]])
            patch_size: size of the patch ([int]), ([t, x, y])
            patch_interval: interval between each patch ([int]), ([t, x, y])
            load_to_memory: whether load data into memory or not (bool)
            transform: function of transformation (function)
            random_patch: sample patch in random or not (bool)
            random_patch_seed: seed for randomness (int)
            algorithm: the algorithm of use (str)
        """
        # check arguments
        if len(patch_size) != 3:
            raise Exception("length of patch_size must be 3")

            # initialize
        self.data_weight = []
        self.root = root

        self.file_paths = os.listdir(os.path.join(self.root))
        self.file_paths.sort()
        self.noisy_data_lists = []
        for i in range(len(self.file_paths)):
            data_path = (os.path.join(root, self.file_paths[i]))
            self.noisy_data_lists.append(data_path)

        self.patch_size = patch_size
        self.random_transform = random_transform
        self.patch_rng = np.random.default_rng(random_patch_seed)
        self.augment_rng = np.random.default_rng(random_patch_seed)

    def __len__(self):
        """Returns the total number of images."""
        return len(self.noisy_data_lists)

    def __getitem__(self, index):
        noisy_data_list = self.noisy_data_lists[index]
        name = self.noisy_data_lists[index]
        name = name[0:-4]
        noisy_image = torch.from_numpy(skio.imread(noisy_data_list).astype(np.float32)).type(torch.FloatTensor)
        mean, std = image_normalization(noisy_image)

        # slicing
        if self.random_transform:
            t_idx = self.patch_rng.integers(0, noisy_image.size()[0] - self.patch_size[0] + 1)
            y_idx = self.patch_rng.integers(0, noisy_image.size()[1] - self.patch_size[1] + 1)
            z_idx = self.patch_rng.integers(0, noisy_image.size()[2] - self.patch_size[2] + 1)

            # input dataset range
            t_range = slice(t_idx, t_idx + self.patch_size[0])
            y_range = slice(y_idx, y_idx + self.patch_size[1])
            z_range = slice(z_idx, z_idx + self.patch_size[2])

            noisy_image = noisy_image[t_range, y_range, z_range]
            noisy_image = random_transform(noisy_image, self.augment_rng, is_rotate=True)

        tempGradMap = get_tempGradMap(noisy_image, self.patch_size[0])
        noisy_image -= mean
        noisy_image /= std

        return noisy_image, tempGradMap, torch.tensor([[t_idx, t_idx + self.patch_size[0]], \
                                          [y_idx, y_idx + self.patch_size[1]],
                                          [z_idx, z_idx + self.patch_size[2]]]), mean, std, name


"""Referenced by SUPPORT version of datafolder
https://github.com/cabooster/SUPPORT
"""
class DataFolder_test_stitch(Dataset):
    def __init__(self, noisy_image, patch_size=[61, 128, 128], patch_interval=[10, 64, 64], load_to_memory=True, \
                 transform=None, random_patch=False, random_patch_seed=0):
        """
        Arguments:
            noisy_image: noisy image stack (Tensor with dimension [t, x, y])
            patch_size: size of the patch ([int]), ([t, x, y])
            patch_interval: interval between each patch ([int]), ([t, x, y])
            load_to_memory: whether load data into memory or not (bool)
            transform: function of transformation (function)
            random_patch: sample patch in random or not (bool)
            random_patch_seed: seed for randomness (int)
        """
        # check arguments
        if len(patch_size) != 3:
            raise Exception("length of patch_size must be 3")
        if len(patch_interval) != 3:
            raise Exception("length of patch_interval must be 3")

        self.patch_size = patch_size
        self.patch_interval = patch_interval
        self.transform = transform
        self.random_patch = random_patch
        self.patch_rng = np.random.default_rng(random_patch_seed)
        self.noisy_image = noisy_image
        self.mean_image, self.std_image = image_normalization(self.noisy_image)

        # generate index
        self.indices = []
        tmp_size = self.noisy_image.size()

        if np.any(tmp_size < np.array(self.patch_size)):
            raise Exception("patch size is larger than data size")

        self.indices = get_coordinate(tmp_size, patch_size, patch_interval)

    def __len__(self):
        return len(self.indices)  # len(self.indices[0]) * len(self.indices[1]) * len(self.indices[2])

    def __getitem__(self, i):
        # slicing
        if self.random_patch:
            idx = self.patch_rng.integers(0, len(self.indices) - 1)
        else:
            idx = i
        single_coordinate = self.indices[idx]

        # input dataset range
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']

        # for stitching dataset range
        noisy_image = self.noisy_image[init_s:end_s, init_h:end_h, init_w:end_w]

        tempGradMap = get_tempGradMap(noisy_image, self.patch_size[0])

        noisy_image = (noisy_image - self.mean_image) / self.std_image

        return noisy_image, tempGradMap, torch.empty(1), single_coordinate


def gen_train_dataloader(root, patch_size, random_transform, batch_size, shuffle=True):
    """
    Generate dataloader for training

    Arguments:
        patch_size: opt.patch_size
        patch_interval: opt.patch_interval
        noisy_data_list: opt.noisy_data

    Returns:
        dataloader_train
    """
    dataset_train = DataFolder(root, patch_size=patch_size, random_transform=random_transform)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=10)

    return dataloader_train

def _compute_zero_padding(kernel_size):
    r"""Utility function that computes zero padding tuple."""
    computed = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]

def random_transform(input, rng, is_rotate=True):
    """
    Randomly rotate/flip the image

    Arguments:
        input: input image stack (Pytorch Tensor with dimension [b, T, X, Y])
        target: targer image stack (Pytorch Tensor with dimension [b, T, X, Y]), can be None
        rng: numpy random number generator

    Returns:
        input: randomly rotated/flipped input image stack (Pytorch Tensor with dimension [b, T, X, Y])
        target: randomly rotated/flipped target image stack (Pytorch Tensor with dimension [b, T, X, Y])
    """
    rand_num = rng.integers(0, 4)  # random number for rotation
    rand_num_2 = rng.integers(0, 2)  # random number for flip

    input = torch.unsqueeze(input, dim=0)

    if is_rotate:
        if rand_num == 1:
            input = torch.rot90(input, k=1, dims=(2, 3))
        elif rand_num == 2:
            input = torch.rot90(input, k=2, dims=(2, 3))
        elif rand_num == 3:
            input = torch.rot90(input, k=3, dims=(2, 3))

    if rand_num_2 == 1:
        input = torch.flip(input, dims=[2])

    return torch.squeeze(input)

def image_normalization(image):
    mean = torch.mean(image)
    std = torch.std(image)

    return mean, std

def get_tempGradMap(image, in_channels):

    image = mean_filter_2d(image, (3, 3))

    tempGrad = torch.zeros_like(image)
    tempGrad[in_channels // 2 + 1:,:,:] = image[in_channels // 2 + 1:,:,:] - image[in_channels // 2:-1,:,:]
    tempGrad[:in_channels // 2,:,:] = image[:in_channels // 2,:,:] - image[1 : in_channels // 2 + 1,:,:]
    tempGrad = mean_filter_2d(torch.abs(tempGrad), (3, 3))

    tempGradMap = torch.zeros_like(tempGrad)

    for i in range(0, in_channels // 2):
        tempGradMap[in_channels // 2 + (i + 1),:] = torch.max(tempGrad[in_channels // 2 + 1 : in_channels // 2 + (i + 2),:], dim=0).values
        tempGradMap[in_channels // 2 - (i + 1),:] = torch.max(tempGrad[in_channels // 2 - (i + 1) : in_channels // 2,:], dim=0).values

    tempGradMap /= 255.0
    return 1 - tempGradMap
