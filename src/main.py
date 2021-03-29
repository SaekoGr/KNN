from dataset import batch_generator
from model import PSPnet


m = PSPnet()
g = batch_generator(32, False)
batch_n = next(g)
print(batch_n)


X, y = next(g)
print(X.shape)

y_pred = m(X)

print(y_pred.shape)
print("DONE!")

