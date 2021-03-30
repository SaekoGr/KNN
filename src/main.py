from dataset import batch_generator, loading
from model import PSPnet
from time import perf_counter
from gc import collect

m = PSPnet()
g = batch_generator(64, 16, False)
batch_n = next(g)
print(batch_n)

s = perf_counter()
for n in range(batch_n):
	X, _ = next(g)
	_ = m(X)
	del X
	loading(n+1, batch_n)
	collect()

print("total time = ", perf_counter() - s)

# print(y_pred.shape)
print("DONE!")

