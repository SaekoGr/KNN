from dataset import batch_generator, loading
from model import PSPnet
from torch import no_grad
import torch
import torch.nn.functional as F
import numpy as np



if torch.cuda.is_available():
  dev = "cuda:0" 
else:
  dev = "cpu"  
device = torch.device(dev)

m = PSPnet()
m.to(device)

lr = 0.001
batch_size = 12
min_res = 16
optimizer = torch.optim.Adam(m.parameters(), lr=lr)
loss_fce = F.binary_cross_entropy

epoch_losses = []
mean_epoch_losses = [] 


# Run 100 epochs
for n in range(100):
	g_train = batch_generator(batch_size, min_res, False, False)
	batch_n = next(g_train)
	# First train model on dataset
	for n in range(batch_n):
		X, y, refs, _ = next(g_train)
		optimizer.zero_grad()
		
		# Add another click of user
		m.add_refinement_map_train(refs)

		# Predict result
		y_pred = m(X)

		loss = loss_fce(y_pred, y)
		loss.backward()
		optimizer.step()

		del X
		del y_pred
		del y
		torch.cuda.empty_cache()
		loading(n+1, batch_n)

		epoch_losses.append(loss.item())
	mean_epoch_losses.append(np.asarray(epoch_losses).mean())

	
	# g_test = batch_generator(batch_size, min_res, False)
	# with no_grad():
	# 	acc = evaluate(m)
	# 	accuracies.append(acc)

print("DONE!")

