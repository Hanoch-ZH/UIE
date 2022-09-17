import numpy as np
from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename
from utils.image_quality_util import getSSIM, getPSNR
from utils.uqim_utils import getUIQM


def SSIMs_PSNRs(gtr_dir, gen_dir, im_res=(256, 256)):

    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    gen_paths = sorted(glob(join(gen_dir, "*.*")))
    ssim_list, psnr_list = [], []
    for gtr_path, gen_path in zip(gtr_paths, gen_paths):
        gtr_f = basename(gtr_path).split('.')[0]
        gen_f = basename(gen_path).split('.')[0]
        if gtr_f == gen_f:

            r_im = Image.open(gtr_path).resize(im_res)
            g_im = Image.open(gen_path).resize(im_res)

            ssim = getSSIM(np.array(r_im), np.array(g_im))
            ssim_list.append(ssim)

            r_im = r_im.convert("L")
            g_im = g_im.convert("L")
            psnr = getPSNR(np.array(r_im), np.array(g_im))
            psnr_list.append(psnr)

    return np.array(ssim_list), np.array(psnr_list)


def UIQMs(dir_name, im_res=(256, 256)):
    paths = sorted(glob(join(dir_name, "*.*")))
    uqims = []
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uiqm = getUIQM(np.array(im))
        uqims.append(uiqm)
    return np.array(uqims)

