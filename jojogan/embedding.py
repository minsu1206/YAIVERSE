"""
from
    https://github.com/minsu1206/BlurFaceDetection/blob/main/experiments/embedding.py
"""

import argparse
from tqdm import tqdm
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from facenet_pytorch import InceptionResnetV1
# from deepface import DeepFace

def load_facenet(device):
	return InceptionResnetV1(pretrained='vggface2').eval().to(device)


def L1_distance(emb1, emb2):
	return np.abs(np.sum(np.array(emb1)-np.array(emb2)))

def L2_distance(emb1, emb2):
	return np.sqrt(np.sum(np.square(np.array(emb1)- np.array(emb2))))

def cos_sim(emb1, emb2):
	return np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
	
def get_n_sort_samples(emb_option, iteration=2000):
	file_path = os.path.join('../examples/samples')
	print("Sorting ...")
	with open(os.path.join(file_path, f'results_{emb_option}.txt'), 'r') as f:
		labels = f.read().splitlines()

	samples = []
	for i in tqdm(range(iteration)):
		samples += [cv2.cvtColor(cv2.imread(os.path.join(file_path, f'sample_{i}.png')), cv2.COLOR_BGR2RGB)]
	samples, labels = np.array(samples), np.array(labels)
	arg = np.argsort(labels)
	samples, labels = samples[arg], labels[arg]
	return samples, labels

def get_facenet_embeddings(samples):
	resnet = InceptionResnetV1(pretrained='vggface2').eval()
	print("Create embeddings ...")
	embs = []
	for blur_img in tqdm(samples):
		embs += [resnet(torch.Tensor(blur_img).permute(2, 0, 1).unsqueeze(0)).squeeze(0).detach().numpy()]
	
	return embs

def get_deepface_embeddings(samples, model_name="VGG-Face"):
	print("Create embeddings ...")
	embs = []
	for blur_img in tqdm(samples):
		embs += [np.array(DeepFace.represent(blur_img, model_name=model_name, enforce_detection=False))]
	return embs

def get_L1_distances(emb_origin, embs):
	print("Calculate distances ...")
	return [L1_distance(emb_origin, emb) for emb in tqdm(embs)]

def get_L2_distances(emb_origin, embs):
	print("Calculate distances ...")
	return [L2_distance(emb_origin, emb) for emb in tqdm(embs)]

def get_cos_similarity(emb_origin, embs):
	print("Calculate similarities ...")
	return [cos_sim(emb_origin, emb) for emb in tqdm(embs)]
