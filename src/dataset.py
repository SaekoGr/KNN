import os, json
import torch
from random import shuffle
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
from clicks import get_maps
from copy import deepcopy



trans = transforms.ToTensor()
transI = transforms.ToPILImage()
margin = 32
margin_half = margin/2

def determine_margins(l_p_max, r_p_max):
	if l_p_max + r_p_max >= margin:
		if r_p_max > l_p_max:
			l_p = -l_p_max if l_p_max <= margin_half else -margin_half
			r_p = margin - l_p
		else:
			r_p = r_p_max if r_p_max <= margin_half else margin_half
			l_p = -(margin - r_p)
	
	else: # There is not enough horizontal padding
		l_p, r_p = 0, 0

	return l_p, r_p
  

def generate_y(x, segmentation):
	# y = Image.fromarray(np.zeros((x.size), dtype="uint8"))
	y = Image.new("L", x.size)


	draw = ImageDraw.Draw(y)
	# draw_test = ImageDraw.Draw(test)
	for polygon in segmentation:
		if type(polygon) == str:
			# y = parse_run_encoding(y, segmentation["counts"])
			break
		else:
			draw.polygon(polygon, outline=255, fill=255)

	return y

def batch_generator(batch_size, min_res_size, train, test, isTrain=True, CUDA=True):
	"""Batch generator for training&testing data.
	if type == train, then generator needs both parts in one folder.
	Path to folder and other detail can be modified in body of function

	NOTE: generator first yields number of batches in data

	Args:
	-----
		batch_size (Integer): Number of data in batch
		isTrain (Bool, optional): Specifying train or test data to parse. Defaults to "train".

	Yields:
	-------
		([torch.Tensor], [torch.Tensor]): Batches of tensors
	"""

	annotation_file = "annotation.json"
	train_folder_path = train
	test_folder_path = test

	if isTrain:
		folder = os.path.join(train_folder_path, "imgs")
		full_path = os.path.join(train_folder_path, annotation_file)
	else:
		folder = os.path.join(test_folder_path, "imgs")
		full_path = os.path.join(test_folder_path, annotation_file)

	with open(full_path) as fd:
		annotation = json.load(fd)

	# if isTrain:
	#   annotation = annotation[:len(annotation)//2]

	batch_n = len(annotation) // batch_size
	yield batch_n

	batch_pool = {}
	x_batch = []
	y_batch = []
	new_bboxes = []
	max_res_size = 176
	while True:
		shuffle(annotation)
		for img_obj in annotation:

			#! Skipping run encoding and microscopic objects
			if type(img_obj["segmentation"]) == dict or img_obj["bbox"][2] < 10 or img_obj["bbox"][3] < 10:
				continue

			bbox = img_obj["bbox"]

			w = bbox[2] // 16
			w_mod = bbox[2] % 16
			w = (w+1)*16 if w_mod > 0 else w*16

			h = bbox[3] // 16
			h_mod = bbox[3] % 16
			h = (h+1)*16 if h_mod > 0 else h*16

			l_p_max = np.round(bbox[0])
			r_p_max = np.round(img_obj["width"] - (bbox[0] + bbox[2]))
			t_p_max = np.round(bbox[1])
			b_p_max = np.round(img_obj["height"] - (bbox[1] + bbox[3]))

			l_p, r_p = determine_margins(l_p_max, r_p_max)
			t_p, b_p = determine_margins(t_p_max, b_p_max)

			w = w if w > min_res_size else min_res_size
			h = h if h > min_res_size else min_res_size

			key = (w,h,l_p,r_p,t_p,b_p)

			if key not in batch_pool:
				batch_pool[key] = [img_obj]
			else:
				batch_pool[key].append(img_obj)

			if len(batch_pool[key]) >= batch_size:
				for img_obj in batch_pool[key]:
					bbox = np.array(img_obj["bbox"])

					x = Image.open(os.path.join(folder, img_obj["file_name"])).convert("RGB")
					segmentation = [list(np.round(polygon)) for polygon in img_obj["segmentation"]]
					y = generate_y(x, segmentation)

					crop_coords = np.round((bbox[0] + l_p, bbox[1] + t_p, bbox[0] + bbox[2] + r_p, bbox[1] + bbox[3] + b_p))
					x = x.crop(crop_coords)
					y = y.crop(crop_coords)

					cropped_size = list(x.size)
					resize_size = deepcopy(cropped_size)
					new_bbox = [-l_p, -t_p, -l_p + bbox[2], -t_p + bbox[3]]

					# print(bbox)
					# print(f"cropped size = {cropped_size}")

					if cropped_size[0] > max_res_size:
						resize_size[0] = max_res_size
					else:
						resize_size[0] = round(w) if w < max_res_size else max_res_size

					if cropped_size[1] > max_res_size:
						resize_size[1] = max_res_size
					else:
						resize_size[1] = round(h) if h < max_res_size else max_res_size

					x = x.resize(resize_size)
					y = y.resize(resize_size)
					ratio = np.divide(resize_size, cropped_size)
					# print("ratio = ", ratio)
					# print("new bbox", new_bbox)
					new_bbox = [ratio[0] * new_bbox[0], ratio[1] * new_bbox[1], ratio[0] * new_bbox[2], ratio[1] * new_bbox[3]]
					# print("new new bbox",new_bbox)

					# Change PIL Image to torch.Tensor
					x_batch.append(trans(x))
					y_batch.append(trans(y))
					new_bboxes.append(new_bbox)

					# Help "print"
					# draw_x = ImageDraw.Draw(x)
					# draw_x.rectangle(new_bbox, outline="red")
					# x.show()
					# y.show()
					# print(img_obj["file_name"])
					# print("category = ", img_obj["category_id"])
					# print(x.size)
					# input()
					# for proc in psutil.process_iter():
					#     if proc.name() == "display":
					#         proc.kill()

				x_batch = torch.stack(x_batch)
				y_batch = torch.stack(y_batch)
				x_batch = get_maps(x_batch, y_batch, new_bboxes)
				if CUDA:
					x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

				yield x_batch, y_batch, new_bboxes
				del batch_pool[key]
				x_batch, y_batch = [], []
				new_bboxes = []


def loading(i, margin):
	dec = i / margin * 10
	print("[{}{}] {:.2f} %".format("*" * int(dec), "_" * int(10-dec), dec * 10))

if __name__ == "__main__":
	batch_size = 5
	min_res_size = 16
	gen = batch_generator(batch_size, min_res_size, True, False)
	l = next(gen)
	print(l)
	for X, y, _, in gen:
		print(X.shape)
		input()

	# from time import perf_counter
	# s = perf_counter()
	# for i in range(l):
	# 	_ = next(gen)
	# 	loading(i+1, l)

	# print("Total time of run is: ", perf_counter() - s)





