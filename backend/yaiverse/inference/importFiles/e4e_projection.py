import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from argparse import Namespace
from yaiverse.inference.importFiles.psp import pSp


@ torch.no_grad()
def projection(img, name, net, device='cuda', save_inverted=False):
    """
    (previous) : load model at here
    (changed) : get loaded model as argument 
    """

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    img = transform(img).unsqueeze(0).to(device)
    # if save_inverted:
    images, w_plus = net(img, randomize_noise=False, return_latents=True)
    # else:
    #     pass

    result_file = {}
    result_file['latent'] = w_plus[0]
    torch.save(result_file, name)

    return w_plus[0]


