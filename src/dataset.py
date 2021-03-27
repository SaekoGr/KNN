import os, json
import torch
from copy import deepcopy
from random import shuffle
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
import psutil

trans = transforms.ToTensor()
transI = transforms.ToPILImage()

def parse_run_encoding(img, coding):
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
			# y = parse_run_encoding(y, segment["counts"])
			break
		else:
			print(polygon)
			print("num of segments {}".format(len(segmentation)))
			draw.polygon(polygon, outline="white", fill="white")
	
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

	x_batch = []
	y_batch = []
	width = []
	height = []
	while True:
		shuffle(annotation)
		for i, img_obj in enumerate(annotation):
			
			#! Skipping run encoding
			if type(img_obj["segmentation"]) == dict:
				continue

			width.append(img_obj["width"])
			height.append(img_obj["width"])

			
				# Load image
			x = Image.open(os.path.join(folder, img_obj["file_name"]))
			# Generate ground truth for loaded image
			y = generate_y(x, img_obj["segmentation"])

			x_batch.append(x)
			y_batch.append(y)

			if len(x_batch) != 0 and len(x_batch) % batch_size == 0:
				width = max(width)
				height = max(height)
				new_size = (width, height)

				for i, (x, y) in enumerate(zip(x_batch, y_batch)):
					old_size = x.size
					# Create 2 copies of blank black image
					resized_x = Image.new("RGB", new_size)
					resized_y = Image.new("RGB", new_size)

					# insert original image inside the new one
					resized_x.paste(x, ((new_size[0]-old_size[0])//2, (new_size[1]-old_size[1])//2))
					resized_y.paste(y, ((new_size[0]-old_size[0])//2, (new_size[1]-old_size[1])//2))

					resized_x.show()
					resized_y.show()
					print(old_size)
					print(img_obj["category_id"])
					print(img_obj["image_id"])

					input()
					for proc in psutil.process_iter():
						if proc.name() == "display":
							proc.kill()

					x_batch[i] = trans(resized_x)
					y_batch[i] = trans(resized_y)
				
				x_batch = torch.stack(x_batch)
				y_batch = torch.stack(y_batch)
				yield x_batch, y_batch
				width, height = [], []
				x_batch, y_batch = [], []




		# for i in range(batch_n):
		# 	start = batch_size*i
		# 	end = batch_size*(i+1)
		# 	ann_slice = annotation[start:end]

		# 	width = max([x["width"] for x in ann_slice])
		# 	height = max([x["height"] for x in ann_slice])
		# 	new_size = (width, height)
		# 	print(new_size)

		# 	for img_obj in ann_slice:


				
		# 		# Load image
		# 		x = Image.open(os.path.join(folder, img_obj["file_name"]))
		# 		old_size = x.size
		# 		# Generate ground truth for loaded image
		# 		y = generate_y(x, img_obj["segmentation"])

		# 		# Create 2 copies of blank black image
		# 		resized_x = Image.new("RGB", new_size)
		# 		resized_y = deepcopy(resized_x)
				
		# 		# insert original image inside the new one
		# 		resized_x.paste(x, ((new_size[0]-old_size[0])//2, (new_size[1]-old_size[1])//2))
		# 		resized_y.paste(y, ((new_size[0]-old_size[0])//2, (new_size[1]-old_size[1])//2))

		# 		# resized_x.show()
		# 		# resized_y.show()


		# 		# convert PIL image to torch.tensor
		# 		resized_x = trans(resized_x)
		# 		resized_y = trans(resized_y)

		# 		x_batch.append(resized_x)
		# 		y_batch.append(resized_y)

		# 	x_batch = torch.stack(x_batch)
		# 	y_batch = torch.stack(y_batch)

		# 	yield x_batch, y_batch
		# 	x_batch, y_batch = [],[]


				

if __name__ == "__main__":
	batch_size = 1
	gen = batch_generator(batch_size, False)
	l = next(gen)
	print(l)
	for X, y in gen:
		print(X.shape)
		input()






