from dataset import batch_generator, loading, transI
from model import IOGnet
from evaluation_main import evaluate
import torch
import torch.nn.functional as F
import numpy as np



if torch.cuda.is_available():
  dev = "cuda:0" 
else:
  dev = "cpu"  
device = torch.device(dev)

m = IOGnet()
m.to(device)

lr = 0.001
batch_size = 1
min_res = 16
optimizer = torch.optim.Adam(m.parameters(), lr=lr)
loss_fce = F.binary_cross_entropy

epoch_losses = []
mean_epoch_losses = [] 
accuracies = []
model_path = "/content/gdrive/MyDrive/KNN/IOGnet.h5"

g_train = batch_generator(batch_size, min_res ,False, False)
batch_n = next(g_train)

for n in range(100):
  epoch_losses = []
  # First train model on dataset
  for i in range(batch_n):
    X, y, _ = next(g_train)
    # print(X.shape)
    optimizer.zero_grad()

    # Predict result
    y_pred = m(X)

    loss = loss_fce(y_pred, y)
    loss.backward()
    optimizer.step()

    del X
    del y_pred
    del y
    torch.cuda.empty_cache()
    loading(i+1, batch_n)
    epoch_losses.append(loss.item())
    print(loss.item())
  mean_epoch_losses.append(np.asarray(epoch_losses).mean())

  # evaluation
  pixel_acc, iou, dice_coeff = evaluate(n, m, batch_size=round(batch_size*1.5), min_res_size=16)
  accuracies.append(iou)

  torch.save({
        'epoch': n,
        'model_state_dict': m.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'mean_loss' : mean_epoch_losses,
        'pixel_acc' : pixel_acc,
        'iou' : iou,
        'dice_coeff' : dice_coeff
        }, f"{model_path}{n}.json")
  
  print(f"iou accuracy of number {n} is {iou}")
  if iou > best_model_acc:
    print(f"New best model is number {n}.")
    with open("/content/gdrive/MyDrive/KNN/Model/best_index.txt", "w") as fd:
      fd.write(str(best_model_index))
    best_model_acc = iou
    best_model_index = n
    
  print(f"best model is {best_model_index}")

print("DONE!")