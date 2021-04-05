from dataset import batch_generator, loading
from model import PSPnet
from time import perf_counter
from gc import collect
from torch import no_grad

m = PSPnet()
g = batch_generator(1, 16, False)
batch_n = next(g)
print(batch_n)

s = perf_counter()
for n in range(batch_n):
	X, _ = next(g)
	print(X.shape)
	with no_grad():
		y_pred = m(X)
	print(y_pred.shape)
	del X
	loading(n+1, batch_n)
	collect()

print("total time = ", perf_counter() - s)

# print(y_pred.shape)
print("DONE!")

