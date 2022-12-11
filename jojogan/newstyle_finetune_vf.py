"""
Modified version 
- from JoJoGAN / stylize.ipynb
- for Pretraining and Finetuning


"""
from copy import deepcopy
import time
import torch
torch.backends.cudnn.benchmark = True
from torchvision import transforms, utils
from torchvision.utils import save_image
from PIL import Image
import math
import random
import os
import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
from model import *
from custom_face_align import face_align as align_face
from e4e.models.psp import pSp as e4e_pSp
from e4e_projection import projection as e4e_projection

import argparse
import sys
sys.path.append(os.getcwd().replace('YAIVERSE/jojogan', 'YAIVERSE/restyle_encoder'))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import restyle_encoder.utils.inference_utils as restyle_utils
from restyle_encoder.models.psp import pSp as restyle_pSp
# TODO
from embedding import *


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

def restyle(net, imgs, n_iters, transforms, style_code_path, device='cuda:0'):
    transformed_imgs = transforms(imgs)
    opt_dict = argparse.Namespace()
    opt_dict.n_iters_per_batch = n_iters
    opt_dict.resize_outputs = False
    opt_dict.dataset_type = "ffhq_encode"
    with torch.no_grad():
        avg_image = restyle_utils.get_average_image(net, opt_dict)
        result_batch, result_latents = restyle_utils.run_on_batch(
            transformed_imgs.to(device).unsqueeze(0), net, opt_dict, avg_image)
    
    # FIXME : save 에서 에러 뜨는데 type 에러 같다
    # torch.save(style_code_path, torch.tensor(result_latents[0][n_iters - 1]))
    # print(result_latents[0][n_iters - 1].shape)
    # return result_latents[0][n_iters - 1]
    last_latent = torch.from_numpy(result_latents[0][n_iters - 1])
    torch.save(last_latent, style_code_path)
    return last_latent

def wo_extension(path):
    if path.endswith('.png'):
        path = path.replace('.png', '')
    if path.endswith('.jpg'):
        path = path.replace('.jpg', '')
    if path.endswith('.jpeg'):
        path = path.replace('.jpeg', '')
    return path

def finetune(args):
    # -------------------------------------------------------------------- #
    #                                   Setting
    # -------------------------------------------------------------------- #
    print("-------------------- Setting --------------------")
    device = args.device
    latent_dim = args.latent_dim
    output_dir = args.output_dir

    # (0) GAN inversion model for face image
    inversion_model_path = f'{args.face_inversion_dir}/e4e_ffhq_encode.pt'
    ckpt = torch.load(inversion_model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = inversion_model_path
    opts= argparse.Namespace(**opts)
    face_inversion_net = e4e_pSp(opts, device).eval().to(device)
    print(f"LOAD GAN inversion model for real face : {args.face_inversion_dir}")

    # (1) GAN inversion model for toon image
    if args.toon_inversion_model == 'e4e':
        toon_inversion_net = face_inversion_net
    elif args.toon_inversion_model == 'restyle':  # recommend
        inversion_model_path = f'{args.toon_inversion_dir}/restyle_psp_ffhq_encode.pt'
        ckpt = torch.load(inversion_model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = inversion_model_path
        opts = argparse.Namespace(**opts)
        toon_inversion_net = restyle_pSp(opts).eval().to(device)
    print(f"LOAD GAN inversion model for toon : {args.toon_inversion_dir}")

    # (1) StyleGAN2 model
    original_generator = Generator(1024, latent_dim, 8, 2).to(device)
    ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    original_generator.load_state_dict(ckpt["g_ema"], strict=False)
    mean_latent = original_generator.mean_latent(10000)
    generator = deepcopy(original_generator)
    mean_latent = original_generator.mean_latent(10000)
    print(f"LOAD StyleGAN2 Generator")
    # load discriminator for perceptual loss
    discriminator = Discriminator(1024, 2).eval().to(device)
    discriminator.load_state_dict(ckpt["d"], strict=False)
    print(f"LOAD StyleGAN2 Discriminator")

    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # (2) StyleGAN setting
    # reset generator
    del generator
    generator = deepcopy(original_generator)
    g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))

    # Which layers to swap for generating a family of plausible real images -> fake image
    if args.preserve_color:
        id_swap = [9,11,15,16,17]
    else:
        id_swap = list(range(7, generator.n_latent))
    

    # (3) (optional) Face Recognition model
    net_face_recog = None
    tf_gray = None
    if args.use_face_recog:
        net_face_recog = load_facenet(device)
        tf_gray = transforms.Compose([transforms.Resize((224,224)),
                                transforms.Grayscale(num_output_channels=3)])

    # -------------------------------------------------------------------- #
    #                                   Inversion
    # -------------------------------------------------------------------- #
    print("----------------- Inversion ------------------")
    aligned_face = align_face(args.img_path)
    name = args.img_path.replace('.png', '_inversion.pt')
    my_w = e4e_projection(
        aligned_face, 
        name,
        net=face_inversion_net, 
        device=device).unsqueeze(0)
    print(f"Get w latent from my face : {args.img_path}")

    style_name = args.style_name
    img_names = []
    for name in args.style_img_name:       # check image extension
        if name.endswith('.jpg') or name.endswith('.png') or name.endswith('.jpeg'):
            img_names.append(name)
    names = [os.path.join(args.style_dir, name) for name in img_names]
    print("Style sources : ", names)
    targets = []
    latents = []
    target_embed = []
    os.makedirs(f'{output_dir}/style_images_aligned/{style_name}',exist_ok=True)
    os.makedirs(f'{output_dir}/inversion_codes/{style_name}',exist_ok=True)

    for name in names:
       
        # crop and align the face
        name_wo_ext = wo_extension(os.path.basename(name))
        style_aligned_path = f'{output_dir}/style_images_aligned/{style_name}/{name_wo_ext}.png'
        if not os.path.exists(style_aligned_path):
            style_aligned = align_face(name)
            style_aligned = Image.fromarray(style_aligned).convert('RGB')
            style_aligned.save(style_aligned_path)
        else:
            style_aligned = Image.open(style_aligned_path).convert('RGB')

        # GAN invert
        style_code_path = f'{output_dir}/inversion_codes/{style_name}/{name_wo_ext}.pt'
        if not os.path.exists(style_code_path):
            if args.toon_inversion_model == 'e4e':
                latent = e4e_projection(style_aligned, style_code_path, toon_inversion_net, device)
            elif args.toon_inversion_model == 'restyle':
                inv_transforms = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                latent = restyle(toon_inversion_net, style_aligned, args.restyle_iter, inv_transforms, style_code_path, device)
        else:
            latent = torch.load(style_code_path)

        # (optional) face embedding
        if net_face_recog != None:
            target_embed.append(net_face_recog(tf_gray(style_aligned).to(device))) 

        targets.append(transform(style_aligned).to(device))
        latents.append(latent.to(device))

    targets = torch.stack(targets, 0)
    latents = torch.stack(latents, 0)
    
    if net_face_recog != None:
        target_embed = torch.stack(target_embed, 0)

    print(f"Get w latent from target toon images")

    # target_im = utils.make_grid(targets, normalize=True, range=(-1, 1))
    # display_image(target_im, title='Style References')

    # -------------------------------------------------------------------- #
    #                                   Finetuning
    # -------------------------------------------------------------------- #

    print("-------------------- FINETUNE StyleGAN --------------------")

    for idx in tqdm(range(args.num_iter)):
        mean_w = generator.get_latent(torch.randn([latents.size(0), args.latent_dim]).to(device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
        in_latent = latents.clone()
        in_latent[:, id_swap] = args.alpha*latents[:, id_swap] + (1-args.alpha)*mean_w[:, id_swap]

        img = generator(in_latent, input_is_latent=True)

        with torch.no_grad():
            real_feat = discriminator(targets)
        fake_feat = discriminator(img)

        id_loss = 0
        if net_face_recog != None:
            img_embed = [net_face_recog(tf_gray(img[i, :, :, :]).unsqueeze(0)) for i in range(img.shape[0])]
            img_embed = torch.stack(img_embed, 0)
            id_loss = F.cosine_similarity(target_embed, img_embed, dim=1).mean()

        loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)])/len(fake_feat)
        loss += id_loss
        # loss should be decreased significantly.
        # check loss graph if the result is not desired.

        g_optim.zero_grad()
        loss.backward()
        g_optim.step()

    # save stylegan_model
    stylegan_model_path = f'{args.stylegan_dir}/{style_name}.pt'
    torch.save({'g': generator.state_dict()},stylegan_model_path)
    print(f'StyleGAN model saved as {stylegan_model_path}!')

    ckpt = torch.load(stylegan_model_path, map_location=device)
    generator = Generator(1024, args.latent_dim, 8, 2).to(device)
    generator.load_state_dict(ckpt["g"], strict=False)


    print("-------------------- GENERATE RESULT --------------------")
    torch.manual_seed(args.seed)

    with torch.no_grad():
        generator.eval()
        z = torch.randn(args.n_sample, args.latent_dim, device=device)
        original_sample = original_generator([z], truncation=0.7, truncation_latent=mean_latent)
        sample = generator([z], truncation=0.7, truncation_latent=mean_latent)
        original_my_sample = original_generator(my_w, input_is_latent=True)
        my_sample = generator(my_w, input_is_latent=True)

    # display reference images

    face = transform(Image.fromarray(aligned_face)).to(device).unsqueeze(0)
    my_output = torch.cat([face, my_sample], 0)
    output = torch.cat([original_sample, sample], 0)

    os.makedirs(f'{args.output_dir}/result', exist_ok=True)
    save_image(
        utils.make_grid(my_output, normalize=True, range=(-1, 1)), 
        f'./{args.output_dir}/result/{style_name}_image_output.png')
    save_image(
        utils.make_grid(output, normalize=True, range=(-1, 1)),
        f'./{args.output_dir}/result/{style_name}_random_output.png')

    print(f'Random output image saved as {args.output_dir}/result/{style_name}_image_output.png!')
    print(f'Output image saved as {args.output_dir}/result/{style_name}_random_output.png!')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default = "cuda:0")
    # args : input / output
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--style_dir', type=str, required=True)
    parser.add_argument('--style_name', type=str,required=True)
    parser.add_argument('--style_img_name', nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    # args : stylegan 
    parser.add_argument('--alpha', type=float, default=0.0) 
    parser.add_argument('--preserve_color', type=bool, default = True)
    parser.add_argument('--num_iter', type=int, default=300)
    parser.add_argument('--log_interval ', type=int, default=50)
    parser.add_argument('--stylegan_dir', type=str, default='models')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_sample', type=int, default=5)
    parser.add_argument('--time_stamp', action="store_true")
    parser.add_argument('--latent_dim', type=int, default = 512)
    # args : inversion
    parser.add_argument('--face_inversion_dir', type=str, default='models')
    parser.add_argument('--toon_inversion_dir', type=str, default='../restyle_encoder/pretrained_models')
    parser.add_argument('--toon_inversion_model', type=str, default='restyle', 
        help='choose one among [e4e, restyle]. recommend restyle')
    parser.add_argument('--restyle_iter', type=int, default=10,
        help='only available when inversion model == restyle')
    # args : additional
    parser.add_argument('--use_face_recog', action='store_true')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    finetune(args)





