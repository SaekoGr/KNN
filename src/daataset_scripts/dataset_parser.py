import os, json


def loading(i, margin):
	dec = i / margin * 10
	print("[{}{}] {:.2f} %".format("*" * int(dec), "_" * int(10-dec), dec * 10), end="\r")


def isPart1(imgs, img):
	return imgs[img]


def main():
	cwd = os.getcwd()
	part1 = {}
	part2 = {}

	print("Working on train dataset")

	file_path = os.path.join(cwd, "../../../dataset/coco/annotations/instances_train2017.json")
	with open(file_path, "r") as fd:
		annotation = json.load(fd)

	print("annotation loaded")
	
	img_len = len(annotation["images"])
	img_division = {}
	toggle = True
	for i, img in enumerate(annotation["images"]):
		if toggle:
			part1[img["id"]] = {"width": img["width"], "height": img["height"], "file_name": img["file_name"], "annotations": []}
			img_division[img["file_name"]] = True
		else:
			part2[img["id"]] = {"width": img["width"], "height": img["height"], "file_name": img["file_name"], "annotations": []}
			img_division[img["file_name"]] = False


		loading(i+1, img_len)
		toggle = not toggle

	print("images loaded")

	ann_len = len(annotation["annotations"])
	for i, ann in enumerate(annotation["annotations"]):
		if ann["image_id"] in part1:
			part1[ann["image_id"]]["annotations"].append(ann)
		else:
			part2[ann["image_id"]]["annotations"].append(ann)
		
		loading(i+1, ann_len)
	
	print("annotations loaded")

	part1_path = os.path.join(cwd, "../../../dataset/coco/divided_dataset/part1/part1_annotation.json")
	part2_path = os.path.join(cwd, "../../../dataset/coco/divided_dataset/part2/part2_annotation.json")

	with open(part1_path, "w") as fd:
		json.dump(part1, fd)

	print("annotation part1 saved")

	with open(part2_path, "w") as fd:
		json.dump(part2, fd)

	print("annotation part2 saved")

	imgdir_path = os.path.join(cwd, "../../../dataset/coco/train2017")
	part1_dirpath = os.path.join(cwd, "../../../dataset/coco/divided_dataset/part1/imgs")
	part2_dirpath = os.path.join(cwd, "../../../dataset/coco/divided_dataset/part2/imgs")
	imgs = os.listdir(imgdir_path)
	len_imgs = len(imgs)
	for i, img in enumerate(imgs):
		old_path = os.path.join(imgdir_path, img)
		if isPart1(img_division, img):
			new_path = os.path.join(part1_dirpath, img)
		else:
			new_path = os.path.join(part2_dirpath, img)
		os.rename(old_path, new_path)
		loading(i+1, len_imgs)

	print("training images moved")
	print("working on validation dataset")

	val_annpath = os.path.join(cwd, "../../../dataset/coco/annotations/instances_val2017.json")

	with open(val_annpath, "r") as fd:
		annotation = json.load(fd)
	

	val_ann = {}
	img_len = len(annotation["images"])
	for i, img in enumerate(annotation["images"]):
		val_ann[img["id"]] = {"width": img["width"], "height": img["height"], "file_name": img["file_name"], "annotations": []}
		loading(i+1, img_len)

	print("validation images loaded")

	ann_len = len(annotation["annotations"])
	for i, ann in enumerate(annotation["annotations"]):
		val_ann[ann["image_id"]]["annotations"].append(ann)		
		loading(i+1, ann_len)

	print("validation annotation loaded")

	val_dirpath = os.path.join(cwd, "../../../dataset/coco/divided_dataset/val/imgs")
	val_annfile = os.path.join(val_dirpath, "annotation.json")
	with open(val_annfile, "w") as fd:
		json.dump(val_ann, fd)

	print("validation annotation saved")

	imgdir_path = os.path.join(cwd, "../../../dataset/coco/val2017")
	imgs = os.listdir(imgdir_path)
	imgs_len = len(imgs)
	for i, img in enumerate(imgs):
		old_path = os.path.join(imgdir_path, img)
		new_path = os.path.join(val_dirpath, img)
		os.rename(old_path, new_path)
		loading(i+1, imgs_len)

	print("validation images moved")
	print("all done")









if __name__ == "__main__":
	main()

