"""
Modified version 
- from JoJoGAN / stylize.ipynb
- for AI server  inference

"""

import torch
import time

start_time = time.time()

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
print("TIME : IMPORT : ", time.time() - start_time)


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


def unit_test(args):

    if args.unit_test == 0:
        test_name = 'Load Inversion Net'
    elif args.unit_test == 1:
        test_name = 'Load StyleGAN'
    elif args.unit_test == 2:
        test_name = 'Face Align'
    elif args.unit_test == 3:
        test_name = 'Forward : inversion'
    elif args.unit_test == 4:
        test_name = 'Forward : StyleGAN'

    print(f"UNIT TEST for {test_name}")
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
    total_elapse = 0
    
    # unit test 0 : load inversion (e4e)
    if args.unit_test >= 0:
        total_elapse = 0
        if args.unit_test == 0:
            test_iter = args.unit_num 
        else:
            test_iter = 1

        for _ in tqdm(range(test_iter)):
            start_load_inv = time.time()
            
            inversion_model_path = f'{args.inversion_dir}/e4e_ffhq_encode.pt'
            ckpt = torch.load(inversion_model_path, map_location='cpu')
            opts = ckpt['opts']
            opts['checkpoint_path'] = inversion_model_path
            opts= argparse.Namespace(**opts)
            inversion_net = pSp(opts, device).eval().to(device)
            
            elapse = time.time() - start_load_inv

            del ckpt
            total_elapse += elapse
        total_elapse /= test_iter
        time_stamp_val(func_name='Load Inversion Net', elapse=total_elapse)

    # unit test 1 : load generator (StyleGAN)
    if args.unit_test >= 1:
        total_elapse = 0
        if args.unit_test == 1:
            test_iter = args.unit_num 
        else:
            test_iter = 1
        
        for _ in tqdm(range(test_iter)):
            start_load_gan = time.time()

            stylegan_model_path = f'{args.stylegan_dir}/{args.style}.pt'
            ckpt = torch.load(stylegan_model_path, map_location=device)
            generator = Generator(1024, latent_dim, 8, 2).to(device)
            generator.load_state_dict(ckpt["g"], strict=False)

            elapse = time.time() - start_load_gan

            del ckpt
            total_elapse += elapse
        total_elapse /= test_iter
        time_stamp_val(func_name='Load StyleGAN', elapse=total_elapse)

    # unit test 2 : load face detector (dlib)

    if args.unit_test >= 2:
        total_elapse = 0
        if args.unit_test == 2:
            test_iter = args.unit_num
        else:
            test_iter = 1

        for _ in tqdm(range(test_iter)):
            start_load_face_detector = time.time()

            predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
            detector = dlib.get_frontal_face_detector()

            elapse = time.time() - start_load_face_detector

            total_elapse += elapse
        total_elapse /= test_iter
        time_stamp_val(func_name='Load FaceDetector', elapse=total_elapse)

    # -------------------------------------------------------------------- #
    #                                 ! Toonify !
    # -------------------------------------------------------------------- #
    print("-------------------- ! Toonify ! --------------------")
    

    # unit test 3 : face detector
    if args.unit_test >=3:
        total_elapse = 0
        total_elapse_det = 0
        total_elapse_lmk = 0
        total_elapse_pp = 0
        if args.unit_test == 3:
            test_iter = args.unit_num 
        else:
            test_iter = 1

        for _ in tqdm(range(test_iter)):
            start_align_face = time.time()

            aligned_face, elapse_det, elapse_lmk, elapse_pp = custom_align_face(img_path, predictor, detector)

            elapse = time.time() - start_align_face
            total_elapse += elapse
            total_elapse_det += elapse_det
            total_elapse_lmk += elapse_lmk
            total_elapse_pp += elapse_pp

        total_elapse /= test_iter
        total_elapse_det /= test_iter
        total_elapse_lmk /= test_iter
        total_elapse_pp /= test_iter
        time_stamp_val(func_name='Forward : Face Detector', elapse=total_elapse)
        if args.unit_test == 3:
            time_stamp_val(func_name='Forward : Face Detector DET', elapse=total_elapse_det)
            time_stamp_val(func_name='Forward : Face Detector LMK', elapse=total_elapse_lmk)
            time_stamp_val(func_name='Forward : Face Detector PP', elapse=total_elapse_pp)

    

    # unit test 4 : e4e inversion
    if args.unit_test >= 4:
        total_elapse = 0
        if args.unit_test == 4:
            test_iter = args.unit_num 
        else:
            test_iter = 1

        for _ in tqdm(range(test_iter)):
            start_inversion = time.time()

            my_w = e4e_projection(
                                aligned_face,
                                name=img_path.replace('.png', '_inversion.pt'),
                                net=inversion_net,
                                save_inverted=False,
                                device=device).unsqueeze(0)
            elapse = time.time() - start_inversion
            total_elapse += elapse
        total_elapse /= test_iter
        time_stamp_val(func_name='Forward : E4E Inversion', elapse=total_elapse)


    # unit test 5 : generator forward
    if args.unit_test == 5:
        total_elapse = 0
        test_count = 0
        test_iter = args.unit_num
        while test_count < args.unit_num:
            start_toonify = time.time()
            my_toonify = generator(my_w, input_is_latent=True)
            elapse = time.time() - start_toonify
            total_elapse += elapse
            test_count += 1
        total_elapse /= test_iter
        time_stamp_val(func_name='Forward : StyleGAN', elapse=total_elapse)

        # save result
        # transform = transforms.ToPILImage()
        # my_toonify = utils.make_grid(my_toonify, normalize=True, range=(-1, 1)).squeeze(0)
        # my_toonify = transform(my_toonify)
        # os.makedirs(os.path.join(args.output_dir, args.col), exist_ok=True)
        # output_path = os.path.join(args.output_dir, args.col, f'_{args.style}_{args.seed}.png')
        # my_toonify.save(output_path)

    print("-------------------- FININSH --------------------")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    # parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--unit_test', type=int, default=0)
    parser.add_argument('--unit_num', type=int, default=50)
    parser.add_argument('--col', type=str, required=True)
    parser.add_argument('--style', type=str, required=True)
    parser.add_argument('--inversion_dir', type=str, default='models')
    parser.add_argument('--stylegan_dir', type=str, default='models')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time_stamp', action="store_true")
    args = parser.parse_args()

    # Inference
    # os.makedirs(args.output_dir, exist_ok=True)
    unit_test(args)

    


