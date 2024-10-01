import numpy as np
import torch
import skimage.io as skio

from tqdm import tqdm
from utils.data_loader import DataFolder_test_stitch
from model.TeD import SwinIR
from utils.util import parse_arguments, imshow, grad_imshow, np_imshow
import torch.nn as nn
import os


def validate(test_dataloader, model, device):
    """
    Validate a model with a test data

    Arguments:
        test_dataloader: (Pytorch DataLoader)
            Should be DatasetFRECTAL_test_stitch!
        model: (Pytorch nn.Module)

    Returns:
        denoised_stack: denoised image stack (Numpy array with dimension [T, X, Y])
    """
    with torch.no_grad():
        model.eval()
        # initialize denoised stack to NaN array.
        denoised_stack = np.zeros(test_dataloader.dataset.noisy_image.shape, dtype=np.float32)

        # stitching denoised stack
        # insert the results if the stack value was NaN
        # or, half of the output volume
        for _, (noisy_image, grad_map, _, single_coordinate) in enumerate(tqdm(test_dataloader, desc="validate")):
            noisy_image = noisy_image.to(device)  # [b, z, y, x]
            grad_map = grad_map.to(device)
            noisy_image_denoised = model(noisy_image, grad_map)

            T = noisy_image.size(1)
            for bi in range(noisy_image.size(0)):
                stack_start_w = int(single_coordinate['stack_start_w'][bi])
                stack_end_w = int(single_coordinate['stack_end_w'][bi])
                patch_start_w = int(single_coordinate['patch_start_w'][bi])
                patch_end_w = int(single_coordinate['patch_end_w'][bi])

                stack_start_h = int(single_coordinate['stack_start_h'][bi])
                stack_end_h = int(single_coordinate['stack_end_h'][bi])
                patch_start_h = int(single_coordinate['patch_start_h'][bi])
                patch_end_h = int(single_coordinate['patch_end_h'][bi])

                stack_start_s = int(single_coordinate['init_s'][bi])
                denoised_stack[stack_start_s + (T // 2), stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                    = noisy_image_denoised[bi].squeeze()[patch_start_h:patch_end_h, patch_start_w:patch_end_w].cpu()

        # change nan values to 0 and denormalize
        denoised_stack = denoised_stack * test_dataloader.dataset.std_image.numpy() + test_dataloader.dataset.mean_image.numpy()
        return denoised_stack


if __name__ == '__main__':
    ########## Change it with your data ##############
    save_name = '240904_SwinIr_6666'
    save_epoch = 151

    model_file = os.path.join("./results/saved_models", save_name, "model_%d.pth" % (save_epoch))

    opt = parse_arguments()

    image_size = [21, 128, 128]
    image_interval = [1, 10, 10]
    batch_size = 180  # lower it if memory exceeds.
    ##################################################

    model = SwinIR(img_size=(image_size[1], image_size[2]),
                   patch_size=opt.patch_size,
                   in_channels=opt.input_frames,
                   out_channels=opt.output_frames,
                   window_size=opt.window_size, depths=opt.rstb_depths,
                   embed_dim=opt.embed_dim,
                   attn_drop_rate=opt.attn_drop_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=[0], output_device=0)
    model.to(device)

    model.load_state_dict(
        torch.load(model_file, map_location="cuda:0"))

    print('Loaded trained model and optimizer weights of epoch {}'.format(opt.epoch - 1))

    test_path = '/data/leewj/[2023-2024]Fluorescence_denoising/testset'
    test_folder = '240816_test'

    file_paths = os.listdir(os.path.join(test_path, test_folder))
    file_paths.sort()

    for i in range(len(file_paths)):
        data_file = (os.path.join(os.path.join(test_path, test_folder), file_paths[i]))
        save_folder = os.path.join("./results/images", save_name,
                                   "Epoch_%d_TestFolder_%s" % (save_epoch, test_folder))
        os.makedirs(save_folder, exist_ok=True)

        output_file = os.path.join(save_folder, file_paths[i])

        demo_tif = torch.from_numpy(skio.imread(data_file).astype(np.float32)).type(torch.FloatTensor)
        demo_tif = demo_tif[:, :, :]

        testset = DataFolder_test_stitch(demo_tif, patch_size=image_size, patch_interval=image_interval)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
        denoised_stack = validate(testloader, model, device)
        T, X, Y = denoised_stack.shape


        np_imshow(np.concatenate((demo_tif[int(T/2),:,:].squeeze().detach().cpu(),denoised_stack[int(T / 2), :, :].squeeze()), axis=1))

        save_image = np.clip(denoised_stack[(opt.input_frames - 1) // 2:-(opt.input_frames - 1) // 2, :, :], 0,
                             255).astype('uint8')
        skio.imsave(output_file, save_image, metadata={'axes': 'TYX'})

    ########## Change it with your data ##############
    save_name = '240904_SwinIr_6666'
    save_epoch = 201

    model_file = os.path.join("./results/saved_models", save_name, "model_%d.pth" % (save_epoch))

    opt = parse_arguments()

    image_size = [21, 128, 128]
    image_interval = [1, 10, 10]
    batch_size = 180  # lower it if memory exceeds.
    ##################################################

    model = SwinIR(img_size=(image_size[1], image_size[2]),
                   patch_size=opt.patch_size,
                   in_channels=opt.input_frames,
                   out_channels=opt.output_frames,
                   window_size=opt.window_size, depths=opt.rstb_depths,
                   embed_dim=opt.embed_dim,
                   attn_drop_rate=opt.attn_drop_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=[0], output_device=0)
    model.to(device)

    model.load_state_dict(
        torch.load(model_file, map_location="cuda:0"))

    print('Loaded trained model and optimizer weights of epoch {}'.format(opt.epoch - 1))

    test_path = '/data/leewj/[2023-2024]Fluorescence_denoising/testset'
    test_folder = '240816_test'

    file_paths = os.listdir(os.path.join(test_path, test_folder))
    file_paths.sort()

    for i in range(len(file_paths)):
        data_file = (os.path.join(os.path.join(test_path, test_folder), file_paths[i]))
        save_folder = os.path.join("./results/images", save_name,
                                   "Epoch_%d_TestFolder_%s" % (save_epoch, test_folder))
        os.makedirs(save_folder, exist_ok=True)

        output_file = os.path.join(save_folder, file_paths[i])

        demo_tif = torch.from_numpy(skio.imread(data_file).astype(np.float32)).type(torch.FloatTensor)
        demo_tif = demo_tif[:, :, :]

        testset = DataFolder_test_stitch(demo_tif, patch_size=image_size, patch_interval=image_interval)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
        denoised_stack = validate(testloader, model, device)
        T, X, Y = denoised_stack.shape
        np_imshow(np.concatenate((demo_tif[int(T/2),:,:].squeeze().detach().cpu(),denoised_stack[int(T / 2), :, :].squeeze()), axis=1))

        save_image = np.clip(denoised_stack[(opt.input_frames - 1) // 2:-(opt.input_frames - 1) // 2, :, :], 0,
                             255).astype('uint8')
        skio.imsave(output_file, save_image, metadata={'axes': 'TYX'})

        ########## Change it with your data ##############
    save_name = '240904_SwinIr_6666'
    save_epoch = 251

    model_file = os.path.join("./results/saved_models", save_name, "model_%d.pth" % (save_epoch))

    opt = parse_arguments()

    image_size = [21, 128, 128]
    image_interval = [1, 10, 10]
    batch_size = 180  # lower it if memory exceeds.
    ##################################################

    model = SwinIR(img_size=(image_size[1], image_size[2]),
                   patch_size=opt.patch_size,
                   in_channels=opt.input_frames,
                   out_channels=opt.output_frames,
                   window_size=opt.window_size, depths=opt.rstb_depths,
                   embed_dim=opt.embed_dim,
                   attn_drop_rate=opt.attn_drop_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=[0], output_device=0)
    model.to(device)

    model.load_state_dict(
        torch.load(model_file, map_location="cuda:0"))

    print('Loaded trained model and optimizer weights of epoch {}'.format(opt.epoch - 1))

    test_path = '/data/leewj/[2023-2024]Fluorescence_denoising/testset'
    test_folder = '240816_test'

    file_paths = os.listdir(os.path.join(test_path, test_folder))
    file_paths.sort()

    for i in range(len(file_paths)):
        data_file = (os.path.join(os.path.join(test_path, test_folder), file_paths[i]))
        save_folder = os.path.join("./results/images", save_name,
                                   "Epoch_%d_TestFolder_%s" % (save_epoch, test_folder))
        os.makedirs(save_folder, exist_ok=True)

        output_file = os.path.join(save_folder, file_paths[i])

        demo_tif = torch.from_numpy(skio.imread(data_file).astype(np.float32)).type(torch.FloatTensor)
        demo_tif = demo_tif[:, :, :]

        testset = DataFolder_test_stitch(demo_tif, patch_size=image_size, patch_interval=image_interval)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
        denoised_stack = validate(testloader, model, device)
        T, X, Y = denoised_stack.shape
        np_imshow(np.concatenate((demo_tif[int(T/2),:,:].squeeze().detach().cpu(),denoised_stack[int(T / 2), :, :].squeeze()), axis=1))

        save_image = np.clip(denoised_stack[(opt.input_frames - 1) // 2:-(opt.input_frames - 1) // 2, :, :], 0,
                             255).astype('uint8')
        skio.imsave(output_file, save_image, metadata={'axes': 'TYX'})
