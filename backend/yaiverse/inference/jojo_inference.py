import torch
from torchvision import transforms, utils
import os
from yaiverse.inference.importFiles.model import *
from yaiverse.inference.importFiles.e4e_projection import projection as e4e_projection
from yaiverse.inference.importFiles.psp import pSp
from yaiverse.inference.face_align_ms import face_align
import argparse
import cv2
from PIL import Image
import mediapipe as mp


pt_without_g = ["disney","disney_preserve_color","jojo", "jojo_preserve_color", "jojo_yasuho", "jojo_yasuho_preserve_color"
                ,"arcane_jinx","arcane_jinx_preserve_color","arcane_caitlyn","arcane_caitlyn_preserve_color", "arcane_multi","arcane_multi_preserve_color","art",
           "sketch_multi"]

class ConvertModel:
    
    """
    Load static models to memory
    """
    def __init__(self):
        self.dir = "/home/yai/backend/"
        self.device = "cuda:0"
        self.latent_dim = 512
        inversion_model_path = self.dir + 'yaiverse/inference/importFiles/e4e_ffhq_encode.pt'
        ckpt = torch.load(inversion_model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = inversion_model_path
        opts= argparse.Namespace(**opts)
        self.inversion_net = pSp(opts, self.device).eval().to(self.device)
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Inference dummy data
        self.generate_face("dummy", "dummy", "sketch_multi")
        
        
    """
    Inference input image with style and save image
    """
    def generate_face(self, col:str,image_col:str, style:str) -> None:
        stylegan_model_path = self.dir + 'yaiverse/inference/models/style_model/{}.pt'.format(style)
        ckpt = torch.load(stylegan_model_path, map_location=self.device)

        generator = Generator(1024, self.latent_dim, 8, 2).to(self.device)
        if style in pt_without_g:
            generator.load_state_dict(ckpt["g"], strict=False)
        else:
            generator.load_state_dict(ckpt, strict=False)
        input_img_path = os.path.join(self.dir + "data", col + '/image.jpg')
        my_w = self.align_inversion(input_img_path)
        
        noise = generator.make_noise()
        my_toonify = generator(my_w, truncation=0.7,input_is_latent=True, noise=noise)
        transform = transforms.ToPILImage()
        my_toonify = utils.make_grid(my_toonify, normalize=True, range=(-1, 1)).squeeze(0)
        my_toonify = transform(my_toonify)
        output_path = os.path.join(self.dir + "data", col + f'/{image_col}.jpg')
        my_toonify.save(output_path)
        
    def generate_face_with_pt(self, col:str,image_col:str, style:str) -> None:
        stylegan_model_path = self.dir + '/yaiverse/inference/models/style_model/{}.pt'.format(style)
        ckpt = torch.load(stylegan_model_path, map_location=self.device)

        generator = Generator(1024, self.latent_dim, 8, 2).to(self.device)
        if style in pt_without_g:
            generator.load_state_dict(ckpt["g"], strict=False)
        else:
            generator.load_state_dict(ckpt, strict=False)
        my_w = self.get_inversion_pt(col)
        
        noise = generator.make_noise()
        my_toonify = generator(my_w, truncation=0.7,input_is_latent=True, noise=noise)
        transform = transforms.ToPILImage()
        my_toonify = utils.make_grid(my_toonify, normalize=True, range=(-1, 1)).squeeze(0)
        my_toonify = transform(my_toonify)
        output_path = os.path.join(self.dir + "data", col + f'/{image_col}.jpg')
        my_toonify.save(output_path)

    """
    Align & crop image
    return inversion vector
    """
    def align_inversion(self, img_path:str):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        aligned_face = Image.fromarray(cv2.cvtColor(face_align(img, mp_face_mesh = self.mp_face_mesh), cv2.COLOR_BGR2RGB))
        
        aligned_face.save(img_path.replace("image.jpg", "aligned.jpg"))
        
        return e4e_projection(
                            aligned_face,
                            name=img_path.replace('image.jpg', 'inversion.pt'),
                            net=self.inversion_net,
                            save_inverted=False,
                            device=self.device).unsqueeze(0)
        
        
    def get_inversion_pt(self, col):
        return torch.load(self.dir+"data/"+col+"/inversion.pt")["latent"].unsqueeze(0)
        
