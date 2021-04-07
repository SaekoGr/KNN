import os, json
import torch
from random import shuffle
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
import psutil
from clicks import get_maps
from math import ceil


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

def batch_generator(batch_size, min_res_size, isTrain=True, CUDA=True):
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
	# train_folder_path = "/home/sabi/Desktop/KNN/part1" #"/mnt/d/Škola/Ing_2020_leto/KNN/Projekt/dataset/coco/divided_dataset/train"
	# test_folder_path = "/home/sabi/Desktop/KNN/val/" #"/mnt/d/Škola/Ing_2020_leto/KNN/Projekt/dataset/coco/divided_dataset/val/"

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
		for img_obj in annotation:

			#! Skipping run encoding and microscopic objects
			if type(img_obj["segmentation"]) == dict or img_obj["area"] < 20 or (img_obj["bbox"][2] > 512 and img_obj["bbox"][3] > 512):
				continue

			# Get nearest bigger power of 2 of width and height
			w = pow(2, ceil(np.log(img_obj["bbox"][2])/np.log(2)))
			h = pow(2, ceil(np.log(img_obj["bbox"][3])/np.log(2)))

			w = w if w > min_res_size else min_res_size
			h = h if h > min_res_size else min_res_size

			if (w,h) not in batch_pool:
				batch_pool[(w,h)] = [img_obj]
			else:
				batch_pool[(w,h)].append(img_obj)
			
			if len(batch_pool[(w,h)]) == batch_size:
				new_size = (w, h)

				for img_obj in batch_pool[(w,h)]:
					bbox = img_obj["bbox"]
					x = Image.open(os.path.join(folder, img_obj["file_name"])).convert("RGB")
					# segmentation = [[round(x) for x in polygon] for polygon in img_obj["segmentation"]]
					segmentation = [list(np.round(polygon)) for polygon in img_obj["segmentation"]]
					y = generate_y(x, segmentation)

					# Calculate how much padding will be needed from bouding box
					x_padding = round((w - round(bbox[2])) // 2)
					x_padding = x_padding if x_padding % 2 == 0 else x_padding - 1
					y_padding = round((h - round(bbox[3])) // 2)
					y_padding = y_padding if y_padding % 2 == 0 else y_padding - 1 

					# Get available paddings from image bounding box
					# And set paddings from from bounding box in image
					img_max_right_margin = (img_obj["width"] - (bbox[0] + bbox[2]))
					img_max_bottom_margin = (img_obj["height"] - (bbox[1] + bbox[3]))
					l_p = bbox[0] if bbox[0] < x_padding else x_padding
					r_p = img_max_right_margin if img_max_right_margin < x_padding else x_padding
					t_p = bbox[1] if bbox[1] < y_padding else y_padding
					b_p = img_max_bottom_margin if img_max_bottom_margin < y_padding else y_padding

					# Get coordinates for cropping window
					# And crop x,y images
					crop_coords = (bbox[0] - l_p, bbox[1] - t_p, bbox[0] + bbox[2] + r_p, bbox[1] + bbox[3] + b_p)
					# print("crop_coords = ", crop_coords)
					x = x.crop(crop_coords)
					y = y.crop(crop_coords)
					old_size = x.size

					# print("old_size = ", old_size)
					# print("new_size = ", new_size)

					if old_size != new_size:
						# For images with insufficent padding reserve
						# Generate blank (black) image and insert cropped image in the middle
						w_m = round((new_size[0]-old_size[0])//2)
						h_m = round((new_size[1]-old_size[1])//2)
						resized_x = Image.new("RGB", new_size)
						resized_x.paste(x, (w_m, h_m))

						resized_y = Image.new("L", new_size)
						resized_y.paste(y, (w_m, h_m))
					else:
						w_m, h_m = 0, 0
						resized_x = x
						resized_y = y


					# print(f"w_m = {w_m}, h_m = {h_m}")
					# print(f"l_p = {l_p}, r_p = {r_p}")
					# print(f"t_p = {t_p}, b_p = {b_p}")
					# print("old bbox = ", bbox)
					# print("new bbox = ", new_bbox)

					new_bbox = tuple(np.round((w_m + l_p, h_m + t_p, w_m + l_p + bbox[2], h_m + t_p + bbox[3])))
					new_bboxes.append(new_bbox)

					

					# Help "print"
					draw_x = ImageDraw.Draw(resized_x)
					draw_x.rectangle(new_bbox, outline="red")
					resized_x.show()
					resized_y.show()
					print(img_obj["file_name"])
					print("category = ", img_obj["category_id"])
					input()
					for proc in psutil.process_iter():
						if proc.name() == "display":
							proc.kill()

					# Change PIL Image to torch.Tensor
					x_batch.append(trans(resized_x))
					y_batch.append(trans(resized_y))
				
				x_batch = torch.stack(x_batch)
				y_batch = torch.stack(y_batch)
				x_batch, refs = get_maps(x_batch, y_batch, new_bboxes)
				if CUDA:
					x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

				if(isTrain):
					yield x_batch, y_batch, refs
				else:
					yield x_batch, y_batch, refs, new_bboxes
				del batch_pool[(w,h)]
				x_batch, y_batch = [], []
				new_bboxes = []


def loading(i, margin):
	dec = i / margin * 10
	print("[{}{}] {:.2f} %".format("*" * int(dec), "_" * int(10-dec), dec * 10), end="\r")

		
if __name__ == "__main__":
	batch_size = 5
	min_res_size = 16
	gen = batch_generator(batch_size, min_res_size, False, False)
	l = next(gen)
	print(l)
	for X, y, _, _ in gen:
		print(X.shape)
		input()
	
	# from time import perf_counter
	# s = perf_counter()
	# for i in range(l):
	# 	_, _ = next(gen)
	# 	loading(i+1, l)
	
	# print("Total time of run is: ", perf_counter() - s)





