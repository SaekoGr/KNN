#!/usr/bin/env python
from model import PSPnet
from dataset import batch_generator







if __name__ == "__main__":
	n_epoch = 5
	batch_size = 64



	n = PSPnet()
	g = batch_generator(batch_size, False)
	batch_n = next(g)
	print(batch_n)

	X, y = next(g)

	pred_y = n(X)


	print("Done")