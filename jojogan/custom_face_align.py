import os
#import dlib
# from PIL import Image
import numpy as np
# import scipy
# import scipy.ndimage
import cv2
import mediapipe as mp
from skimage import transform as trans
from scipy.ndimage.filters import gaussian_filter

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 255, 255)]

def get_src_landmarks(topleft_x, topleft_y, lmk):
	"""
	x0, x1, y0, y1: (smoothed) bbox coord.
	pnts: landmarks predicted by MTCNN
	"""    

	x = topleft_x
	y = topleft_y
	src_landmarks = [(int(lmk[i][0] - x), int(lmk[i][1] - y)) for i in range(5)]

	return src_landmarks

def get_tar_landmarks(img_size, scale=1, trans_x=0, trans_y=0.1):
	"""    
	img: detected face image
	"""         

	if 0 >= scale or scale > 1:
		raise ValueError("INVALID SCALE")
	if np.abs(trans_x) > 1 or np.abs(trans_y) > 1:
		raise ValueError("INVALID TRANSLATE")

	default = [
		(0.70405826 - trans_x, 0.68193543 - trans_y),	# left eye
		(0.29594174 - trans_x, 0.68193543 - trans_y),	# right eye
		(0.50000000 - trans_x, 0.46852191 - trans_y),	# nose	= center
		(0.36096748 - trans_x, 0.29023552 - trans_y),	# right lib
		(0.63903252 - trans_x, 0.29023552 - trans_y)	# left lib
	]		# from mediapipe canonical mesh

	center_x = default[2][0]
	center_y = default[2][1]
	ratio_landmarks = [
		(center_x + (default[0][0] - center_x) * scale, center_y + (default[0][1] - center_y) * scale),	# left eye
		(center_x + (default[1][0] - center_x) * scale, center_y + (default[1][1] - center_y) * scale),	# right eye
		(0.50000000 , 0.46852191 ),	# nose
		(center_x + (default[3][0] - center_x) * scale, center_y + (default[3][1] - center_y) * scale),	# right lib
		(center_x + (default[4][0] - center_x) * scale, center_y + (default[4][1] - center_y) * scale)	# left lib
	]		# from mediapipe canonical mesh

	tar_landmarks = [[int((xy[0])*img_size[0]), 
					  int((1-xy[1])*img_size[1])] for xy in ratio_landmarks]

	return tar_landmarks
	
# -----------------------------------------------------------------------------

def get_face_mesh(image, size=1024, visualize=False):
	h, w, _ = image.shape
	assert h == w, "DO square_padding first!"
	mp_face_mesh = mp.solutions.face_mesh
	mp_drawing = mp.solutions.drawing_utils 
	mp_drawing_styles = mp.solutions.drawing_styles

	# Run MediaPipe Face Mesh.
	with mp_face_mesh.FaceMesh(
		static_image_mode=True,
		refine_landmarks=True,
		max_num_faces=1,
		min_detection_confidence=0.5) as face_mesh:

		results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		if not results.multi_face_landmarks:
			raise ValueError("NO FACE DETECTED")
		
		mesh = results.multi_face_landmarks[0]		# single face
		# Get landmarks array
		x = [landmark.x * h for landmark in mesh.landmark]
		y = [landmark.y * h for landmark in mesh.landmark]
		lmks = np.column_stack((x, y))											# (478, 2)

		# main landmarks' index of face mesh
		# left eye = 473 th // right eye = 468 th //nose = 1 th // right lib = 78 th // left lib = 308 th
		# landmark_result = [leye, reye, nose, right_lib, left_lib]
		landmark_result = lmks[[473, 468, 1, 78, 308], :]
		# print(landmark_result)
		# Get bounding box from face_mesh - use outer landmarks
		top_left_x, top_left_y = np.min(lmks, axis=0)
		btn_right_x, btn_right_y = np.max(lmks, axis=0)
		bbox_list = [int(top_left_x), int(top_left_y), int(btn_right_x), int(btn_right_y)]
		# print(bbox_list[2] - bbox_list[0], bbox_list[3] - bbox_list[1])

	if visualize:
		for face_landmarks in results.multi_face_landmarks:
			mp_drawing.draw_landmarks(
				image=annotated_image,
				landmark_list=face_landmarks,
				connections=mp_face_mesh.FACEMESH_TESSELATION,
				landmark_drawing_spec=None,
				connection_drawing_spec=mp_drawing_styles
				.get_default_face_mesh_tesselation_style())
			mp_drawing.draw_landmarks(
				image=annotated_image,
				landmark_list=face_landmarks,
				connections=mp_face_mesh.FACEMESH_CONTOURS,
				landmark_drawing_spec=None,
				connection_drawing_spec=mp_drawing_styles
				.get_default_face_mesh_contours_style())
			mp_drawing.draw_landmarks(
				image=annotated_image,
				landmark_list=face_landmarks,
				connections=mp_face_mesh.FACEMESH_IRISES,
				landmark_drawing_spec=None,
				connection_drawing_spec=mp_drawing_styles
				.get_default_face_mesh_iris_connections_style())

		annotated_image = cv2.resize(annotated_image, (size, size))
		
		return annotated_image, landmark_result, bbox_list
	else:
		return landmark_result, bbox_list

# -----------------------------------------------------------------------
# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, scale=0.6, mode='arcface'):
	assert lmk.shape == (5, 2)
	tform = trans.SimilarityTransform()
	lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
	min_M = []
	min_index = []
	min_error = float('inf')
	scale = 0.8
	src = get_tar_landmarks((image_size, image_size), scale=scale)
	src = np.array(src).reshape(5, 2)

	for i in range(len(src)):
		tform.estimate(lmk, src)
		M = tform.params[0:2, :]
		results = np.dot(M, lmk_tran.T)
		results = results.T
		error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
		if error < min_error:
			min_error = error
			min_M = M
			min_index = i

	return min_M, min_index

def norm_crop(img, landmark, scale=0.6, image_size=112, mode='arcface'):
	M, pose_index = estimate_norm(landmark, image_size, mode=mode, scale=scale)
	warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
	return warped


def face_align(img, scale=0.6, visualize=False):
	if type(img) == str:
		if os.path.exists(img):
			img = cv2.imread(img)
		else: 
			raise ValueError("INVALID PATH")
	square = square_padding(img)

	if visualize:
		annotated, lmk, bbox = get_face_mesh(square)
		cv2.imshow('annotated', annotated)
		cv2.waitKey(0)
	else:
		lmk, bbox = get_face_mesh(square)
	
	H, W, _ = img.shape

	# reflection padding : for square shape
	square_x_pad = 0 if max(H, W) == W else int((max(H, W) - W) / 2)
	square_y_pad = 0 if max(H, W) == H else int((max(H, W) - H) / 2)
	img = np.pad(img, ((int(square_y_pad), int(square_y_pad)), (int(square_x_pad), int(square_x_pad)), (0, 0)), 'reflect')

	H, W, _ = img.shape
	h = bbox[2] - bbox[0]
	w = bbox[3] - bbox[1]
	HW = max(H, W) * max(H, W)
	hw = h * w
	scale = min(h/H, w/W)

	# reflection padding : for transformation
	inv_scale = 1 - scale
	x_pad = inv_scale / 2 * w
	y_pad = inv_scale / 2 * h

	img = np.pad(img, ((int(y_pad), int(y_pad)), (int(x_pad), int(x_pad)), (0, 0)), 'reflect')
	lmk[:, 0] += x_pad
	lmk[:, 1] += y_pad
	H, W, _ = img.shape
	img[:, :square_x_pad + int(x_pad)] = gaussian_filter(img[:, :square_x_pad + int(x_pad)], sigma=7)
	img[:, square_x_pad + W + int(x_pad):] = gaussian_filter(img[:, square_x_pad + W +int(x_pad):], sigma=7)
	img[:square_y_pad + int(y_pad), :] = gaussian_filter(img[:square_y_pad + int(y_pad), :], sigma=7)
	img[square_y_pad + H + int(y_pad):, :] = gaussian_filter(img[square_y_pad + H + int(y_pad):, :], sigma=7)
	scale = hw / HW
	
	scale = 0.65

	aligned = norm_crop(img, lmk, scale=0.6 + 0.1 * scale, image_size=512, mode='mediapipe')
	
	aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

	return aligned

def square_padding(image):
	h, w = image.shape[:2]
	maxhw = np.max([h, w])
	pad_h = int((maxhw - h) / 2)
	pad_v = int((maxhw - w) / 2)
	return cv2.copyMakeBorder(image, pad_h, maxhw - h - pad_h, pad_v, maxhw - w - pad_v, cv2.BORDER_CONSTANT)


if __name__ == "__main__":

	# --------- IMAGE  INFERNCE ------------------------------------------------------------------ 
	
	# img_name = 'iu'
	# suffix = ''
	# for ext in ['.png', '.jpg']:
	# 	if os.path.exists(img_name + ext):
	# 		suffix = ext
	# 		break
	# print(img_name + suffix)
	# img = cv2.imread(img_name + suffix, cv2.IMREAD_COLOR)
	# aligned = face_align(img)
	# cv2.imwrite(img_name + suffix.replace('.', '_aligned.'), aligned)

	# cv2.imshow('aligned', aligned)
	# cv2.waitKey(0)

	# --------- FOLDER INFEERNCE ------------------------------------------------------------------ 

	import glob
	root_folder_dir = ''
	character_name = ['video3']
	for character in character_name:
		char_folder = os.path.join(root_folder_dir, character)
		print(char_folder)

		images = list(glob.glob(char_folder + '/*.jpg')) + list(glob.glob(char_folder + '/*.png'))

		for image_path in images:
			if '_aligned' in image_path:
				continue
			img = cv2.imread(image_path)
			try:
				aligned = face_align(img)
				write_name = image_path.replace('.', '_aligned.')
				if os.path.exists(write_name):
					write_name = image_path.replace('.', '_aligned_2.')
				cv2.imwrite(write_name, aligned)
			except:
				print("NOT DETECTED : ", image_path)
				pass

	# --------- VIDEO  INFERNCE ------------------------------------------------------------------ 
	
	# cam = cv2.VideoCapture(0)
	# fps = cam.get(cv2.CAP_PROP_FPS)

	# width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
	# height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	# save_path = 'video_0_6'
	# os.makedirs(save_path, exist_ok=True)
	# writer = cv2.VideoWriter(f'{save_path}/output.mp4', fourcc, fps, (width, height))

	# frame = 0
	# while True:

	# 	ret, img = cam.read()

	# 	if ret == False:
	# 		break
		
	# 	cv2.imwrite(f'{save_path}/{frame}.jpg', img)
		
	# 	aligned = face_align(img)
	# 	aligned = np.uint8(aligned)

	# 	cv2.imwrite(f'{save_path}/{frame}_aligned.jpg', aligned)
	

	# 	if cv2.waitKey(1)&0xFF == 27:
	# 		break
		
	# 	frame += 1

	# cam.release()
	# cv2.destroyAllWindows()
