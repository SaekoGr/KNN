import torch
from model import IOGnet
from dataset import batch_generator
from matplotlib import pyplot as plt

m = IOGnet()
path = "../model/IOGnet_final_12.json"
checkpoint = torch.load(path, map_location=torch.device('cpu'))
m.load_state_dict(checkpoint['model_state_dict'])
print(checkpoint["mean_loss"])
print(checkpoint["iou"])
print(checkpoint["pixel_acc"])
print(checkpoint["dice_coeff"])


# print(checkpoint["mean_loss"])
g = batch_generator(1, 16,False, False)
print("batch_n = ", next(g))

_, axs = plt.subplots(5,6)

for i in range(5):
    for j in range(6):
        axs[i,j].axis("off")

axs[0, 0].set_title('X')
axs[0, 1].set_title('bbox')
axs[0, 2].set_title('clicks')
axs[0, 3].set_title('y')
axs[0, 4].set_title('pred_y')
axs[0, 5].set_title('threshold')


for i in range(5):

    X, y, _ = next(g)
    with torch.no_grad():
        pred_y = m(X)
    
    bbox = X[0][3].squeeze()
    clicks = X[0][4].squeeze()
    X = X[0][:3].permute(1,2,0)
    y = y[0].squeeze()
    pred_y_th = (pred_y[0] > 0.5).float().squeeze()
    pred_y = pred_y[0].squeeze()

    print(i)
    axs[i, 0].imshow(X)
    axs[i, 1].imshow(bbox)
    axs[i, 2].imshow(clicks)
    axs[i, 3].imshow(y)
    axs[i, 4].imshow(pred_y)
    axs[i, 5].imshow(pred_y_th)

plt.tight_layout(pad=0.3, h_pad=0.3)
plt.show()


