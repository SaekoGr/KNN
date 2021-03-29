import os, json
import torch
from random import shuffle
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
import psutil
from clicks import get_maps

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

	batch_pool = {}
	x_batch = []
	y_batch = []
	new_bboxes = []
	while True:
		shuffle(annotation)
		for i, img_obj in enumerate(annotation):

			#! Skipping run encoding
			if type(img_obj["segmentation"]) == dict:
				continue

			w = img_obj["bbox"][2] // 100
			h = img_obj["bbox"][3] // 100

			if (w,h) not in batch_pool:
				batch_pool[(w,h)] = [img_obj]
			else:
				batch_pool[(w,h)].append(img_obj)
			
			if len(batch_pool[(w,h)]) == batch_size:
				width = round(max(x["bbox"][2] for x in batch_pool[(w,h)]))
				height = round(max(x["bbox"][3] for x in batch_pool[(w,h)]))

				# Width and height must be even size
				width = width if width % 2 == 0 else width + 1
				height = height if height % 2 == 0 else height + 1
				margin = 20
				new_size = (width + 2*margin, height + 2*margin)

				for j, img_obj in enumerate(batch_pool[(w,h)]):
					bbox = img_obj["bbox"]
					x = Image.open(os.path.join(folder, img_obj["file_name"])).convert("RGB")
					segmentation = [[round(x) for l, x in enumerate(polygon)] for polygon in img_obj["segmentation"]]
					y = generate_y(x, segmentation)

					# Get available margins from image bounding box
					img_right_margin = (img_obj["width"] - (bbox[0] + bbox[2]))
					img_bottom_margin = (img_obj["height"] - (bbox[1] + bbox[3]))
					l_w = bbox[0] if bbox[0] < margin else margin
					r_w = img_right_margin if img_right_margin < margin else margin
					t_h = bbox[1] if bbox[1] < margin else margin
					b_h = img_bottom_margin if img_bottom_margin < margin else margin

					# Get coordinates for cropping window
					crop_w_start = bbox[0] - l_w
					crop_h_start = bbox[1] - t_h
					crop_w_end = bbox[0] + bbox[2] + r_w
					crop_h_end = bbox[1] + bbox[3] + b_h
					x = x.crop((crop_w_start, crop_h_start, crop_w_end, crop_h_end))
					y = y.crop((crop_w_start, crop_h_start, crop_w_end, crop_h_end))
					old_size = x.size

					# Normalize polygons in segmentation to bounding box
					# segmentation = [[round(x-bbox[0] + l_w) if l%2==0 else round(x-bbox[1] + t_h) for l, x in enumerate(polygon)] for polygon in img_obj["segmentation"]]
					# y = generate_y(x, segmentation)

					x_padding = (width + 2*margin - old_size[0]) // 2
					y_padding = (height + 2*margin - old_size[1]) // 2

					# Create new blank image and paste cropped image onto it
					resized_x = Image.new("RGB", (width + 2*margin, height + 2*margin))
					resized_x.paste(x, ((new_size[0]-old_size[0])//2, (new_size[1]-old_size[1])//2))

					resized_y = Image.new("L", (width + 2*margin, height + 2*margin))
					resized_y.paste(y, ((new_size[0]-old_size[0])//2, (new_size[1]-old_size[1])//2))

					# print("category = {}".format(img_obj["category_id"]))
					# print("id = {}".format(img_obj["file_name"]))
					# print(l_w, r_w)
					# print(t_h, b_h)

					new_bbox = (x_padding + l_w, y_padding + t_h, new_size[0]-x_padding-r_w, new_size[1]-y_padding-b_h)
					new_bboxes.append(new_bbox)

					

					# Help "print"
					# draw_x = ImageDraw.Draw(resized_x)
					# draw_x.rectangle((x_padding + l_w, y_padding + t_h, new_size[0]-x_padding-r_w, new_size[1]-y_padding-b_h), outline="red")
					# resized_x.show()
					# resized_y.show()
					# print(img_obj["file_name"])
					# input()
					# for proc in psutil.process_iter():
					# 	if proc.name() == "display":
					# 		proc.kill()

					# Change PIL Image to torch.Tensor
					x_batch.append(trans(resized_x))
					y_batch.append(trans(resized_y))
				
				x_batch = torch.stack(x_batch)
				y_batch = torch.stack(y_batch)
				x_batch = get_maps(x_batch, y_batch, new_bboxes)
				yield x_batch, y_batch
				del batch_pool[(w,h)]
				x_batch, y_batch = [], []
				new_bboxes = []


def loading(i, margin):
	dec = i / margin * 10
	print("[{}{}] {:.2f} %".format("*" * int(dec), "_" * int(10-dec), dec * 10), end="\r")

		
if __name__ == "__main__":
	batch_size = 64
	gen = batch_generator(batch_size, False)
	l = next(gen)
	print(l)
	# for X, y in gen:
	# 	print(X.shape)
	# 	input()
	
	# from time import perf_counter
	# s = perf_counter()
	# for i in range(l):
	# 	_, _ = next(gen)
	# 	loading(i+1, l)
	
	# print("Total time of run is: ", perf_counter() - s)





