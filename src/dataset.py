import os, json
import torch
from random import shuffle
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
import psutil
from copy import deepcopy

trans = transforms.ToTensor()
transI = transforms.ToPILImage()

def parse_run_encoding(img, coding):
	print("PARSING RUN ENCODING")
	isWhite = False
	index = 0

	a = np.zeros(np.prod(img.size), dtype="uint8")
	for n in coding:
		if isWhite:
			a[index:index+n] = 255
		index += n
		isWhite = not isWhite
	a = np.reshape(a, img.size[::-1])

	return Image.fromarray(a)


def generate_y(x, segmentation):
	# y = Image.fromarray(np.zeros((x.size), dtype="uint8"))
	y = Image.new("RGB", x.size)


	draw = ImageDraw.Draw(y)
	# draw_test = ImageDraw.Draw(test)
	for polygon in segmentation:
		if type(polygon) == str:
			# y = parse_run_encoding(y, segmentation["counts"])
			break
		else:
			draw.polygon(polygon, outline="white", fill="white")
	
	return y.convert('L')

def get_crop_res(bboxes, width, height, max_margin=20):
	l_width = min([x[0] for x in bboxes])
	l_height = min([x[1] for x in bboxes])
	r_width = min([w - (box[0] + box[2]) for w, box in zip(width, bboxes)])
	r_height = min([h - (box[1] + box[3]) for h, box in zip(height, bboxes)])

	# width = max(width)
	# height = max(height)
	width = max([box[2] for box in bboxes])
	height = max([box[3] for box in bboxes])


	max_width_margin = min(l_width, r_width)
	max_height_margin = min(l_height, r_height)

	width_margin =  min(max_margin, max_width_margin)
	height_margin =  min(max_margin, max_height_margin)

	return (width + 2*width_margin, height + 2*height_margin), width_margin, height_margin


def batch_generator(batch_size, isTrain=True):
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
	train_folder_path = "/mnt/d/Škola/Ing_2020_leto/KNN/Projekt/dataset/coco/divided_dataset/train"
	test_folder_path = "/mnt/d/Škola/Ing_2020_leto/KNN/Projekt/dataset/coco/divided_dataset/val/"


	if isTrain:
		folder = os.path.join(train_folder_path, "imgs")
		full_path = os.path.join(train_folder_path, annotation_file)
	else:
		folder = os.path.join(test_folder_path, "imgs")
		full_path = os.path.join(test_folder_path, annotation_file)

	with open(full_path) as fd:
		annotation = json.load(fd)

	batch_n = len(annotation) // batch_size
	yield batch_n

	x_batch = []
	y_batch = []
	width = []
	height = []
	bboxes = []
	while True:
		shuffle(annotation)
		for i, img_obj in enumerate(annotation):
			
			#! Skipping run encoding
			if type(img_obj["segmentation"]) == dict:
				continue

			
				# Load image
			x = Image.open(os.path.join(folder, img_obj["file_name"])).convert("RGB")
			# Generate ground truth for loaded image
			y = generate_y(x, img_obj["segmentation"])

			x_batch.append(x)
			y_batch.append(y)

			bboxes.append(img_obj["bbox"])
			width.append(img_obj["width"])
			height.append(img_obj["height"])

			if len(x_batch) != 0 and len(x_batch) % batch_size == 0:
				crop_window_size, width_margin, height_margin = get_crop_res(bboxes, width, height)
				new_bboxes = []
				for j, (x, y, b) in enumerate(zip(x_batch, y_batch, bboxes)):
					# print(x.size)
					# print(crop_window_size, width_margin, height_margin)
					# print(b)
					# x.show()
					w_l = b[0]-width_margin
					h_t = b[1]-height_margin
					crop_window = (w_l, h_t, w_l + crop_window_size[0], h_t + crop_window_size[1])
					crop_window_size = [round(x) for x in crop_window_size]
					
					# width & height must be even for model reasons
					if crop_window_size[0] % 2 != 0:
						crop_window_size[0] += 1
					if crop_window_size[1] % 2 != 0:
						crop_window_size[1] += 1

					x = x.crop(crop_window).resize(crop_window_size)
					y = y.crop(crop_window).resize(crop_window_size)

					# Holds information about where is cropped object and it's bounding box
					new_bbox = (width_margin, height_margin, width_margin + b[2], height_margin + b[3])
					print(x.size)
					# print(b)
					print(new_bbox)
					new_bboxes.append(new_bbox)

					# draw_x = ImageDraw.Draw(x)
					# draw_y = ImageDraw.Draw(x)
					# draw_x.rectangle(new_bbox, outline="red")
					# draw_y.rectangle(new_bbox, outline=1)
					# print("category id = {}".format(annotation[i-batch_size+j+1]["category_id"]))
					# print("image id = {}".format(annotation[i-batch_size+j+1]["file_name"]))
					# x.show()
					# y.show()
					# input()
					# for proc in psutil.process_iter():
					# 	if proc.name() == "display":
					# 		proc.kill()

					x_batch[j] = trans(x)
					y_batch[j] = trans(y)
				
				x_batch = torch.stack(x_batch)
				y_batch = torch.stack(y_batch)
				yield x_batch, y_batch
				width, height = [], []
				x_batch, y_batch = [], []
				bboxes = []

				
from time import perf_counter
if __name__ == "__main__":
	batch_size = 4
	gen = batch_generator(batch_size, False)
	l = next(gen)
	print(l)
	for X, y in gen:
		# print(X.shape)
		input()





	# annotation_file = "annotation.json"
	# test_folder_path = "/mnt/d/Škola/Ing_2020_leto/KNN/Projekt/dataset/coco/divided_dataset/val/"
	# full_path = os.path.join(test_folder_path, annotation_file)
	# folder = os.path.join(test_folder_path, "imgs")

	# with open(full_path) as fd:
	# 	annotation = json.load(fd)
	
	# for ann in annotation:
	# 	if type(ann["segmentation"]) == dict:
	# 		print(ann["file_name"])
	# 		print(ann["category_id"])
	# 		img = Image.open(os.path.join(folder, ann["file_name"]))
	# 		y = Image.new("RGB", img.size)
	# 		y = generate_y(y, ann["segmentation"])
	# 		img.show()
	# 		y.show()
	# 		input()






