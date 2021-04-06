from dataset import batch_generator
from model import PSPnet
from time import perf_counter
from torch import no_grad
import torch

if torch.cuda.is_available():
  dev = "cuda:0" 
else:
  dev = "cpu"  
device = torch.device(dev)



m = PSPnet()
m.to(device)
g = batch_generator(1, 16, False, False)
batch_n = next(g)
# print(batch_n)
X, y, _ = next(g)
print(X.shape)
with no_grad():
	y_pred = m(X)
print(y.shape)




# s = perf_counter()
# for n in range(batch_n):
# 	X, _ = next(g)
# 	print(X.shape)
# 	with no_grad():
# 		y_pred = m(X)
# 	print(y_pred.shape)
# 	del X
# 	del y_pred
# 	torch.cuda.empty_cache()
# 	loading(n+1, batch_n)

# print("total time = ", perf_counter() - s)

# print(y_pred.shape)
print("DONE!")

