"""
Modified version 
- from JoJoGAN / stylize.ipynb
- for AI server  inference

"""
import time
import torch
torch.backends.cudnn.benchmark = True
from torchvision import transforms, utils
from PIL import Image
import math
import random
import os
import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
from model import *
from util import *
from e4e.models.psp import pSp
from e4e_projection import projection as e4e_projection

import argparse


def time_stamp(func_name, start, template1='TIME', return_val=False):
    stamp = round(time.time() - start, 5)
    print(f'{template1} : {func_name} : {stamp} s')
    if return_val:
        return stamp


def time_stamp_val(func_name, elapse, template1='TIME', return_val=False):
    elapse = round(elapse, 5)
    print(f'{template1} : {func_name} : {elapse} s')
    if return_val:
        return elapse



def inference(args):

    # -------------------------------------------------------------------- #
    #                                   Filter
    # -------------------------------------------------------------------- #
    print("-------------------- Filter --------------------")
    # (0) check : style name - invalid or not
    if args.style not in ['disney', 'jojo', 'arcane', 'art']:
        raise NotImplementedError("NOT SUPPROTED STYLE IS GIVEN")

    # (1) check : img path - invalid or not
    img_path = os.path.join(args.input_dir, args.col, 'image.png')
    if not os.path.exists(img_path):
        raise ValueError("NO INPUT IMG : CHECK YOUR IMG PATH")



    # -------------------------------------------------------------------- #
    #                                   Setting
    # -------------------------------------------------------------------- #
    print("-------------------- Setting --------------------")

    device = "cuda:0"
    latent_dim = 512
    
    start_load_inv = time.time()
    inversion_model_path = f'{args.inversion_dir}/e4e_ffhq_encode.pt'
    ckpt = torch.load(inversion_model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = inversion_model_path
    opts= argparse.Namespace(**opts)
    inversion_net = pSp(opts, device).eval().to(device)
    if args.time_stamp:
        time_stamp(func_name='Load Inversion Net', start=start_load_inv)

    del ckpt

    start_load_gan = time.time()
    stylegan_model_path = f'{args.stylegan_dir}/{args.style}.pt'
    ckpt = torch.load(stylegan_model_path, map_location=device)

    generator = Generator(1024, latent_dim, 8, 2).to(device)
    generator.load_state_dict(ckpt["g"], strict=False)
    if args.time_stamp:
        time_stamp(func_name='Load StyleGAN', start=start_load_gan)



    # -------------------------------------------------------------------- #
    #                                 ! Toonify !
    # -------------------------------------------------------------------- #
    print("-------------------- ! Toonify ! --------------------")
    

    # (0) align_face : dlib face detector
    start_align_face = time.time()
    aligned_face = align_face(img_path)
    if args.time_stamp:
        time_stamp(func_name='align face', start=start_align_face)


    # (1) inversion : e4e
    start_inversion = time.time()
    my_w = e4e_projection(
                        aligned_face,
                        name=img_path.replace('.png', '_inversion.pt'),
                        net=inversion_net,
                        save_inverted=False,
                        device=device).unsqueeze(0)
    if args.time_stamp:
        time_stamp(func_name='E4E inversion', start=start_inversion)


    # (2) generator : finetuned for toonify
    start_toonify = time.time()
    my_toonify = generator(my_w, input_is_latent=True)
    if args.time_stamp:
        time_stamp(func_name='Toonify', start=start_toonify)


    # (3) save result
    transform = transforms.ToPILImage()
    my_toonify = utils.make_grid(my_toonify, normalize=True, range=(-1, 1)).squeeze(0)
    my_toonify = transform(my_toonify)
    output_path = os.path.join(args.output_dir, args.col + f'_{args.style}_{args.seed}.png')
    my_toonify.save(output_path)

    print("-------------------- FININSH --------------------")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--col', type=str, required=True)
    parser.add_argument('--style', type=str, required=True)
    parser.add_argument('--inversion_dir', type=str, default='models')
    parser.add_argument('--stylegan_dir', type=str, default='models')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time_stamp', action="store_true")
    args = parser.parse_args()

    # Inference
    os.makedirs(args.output_dir, exist_ok=True)
    inference(args)

    


