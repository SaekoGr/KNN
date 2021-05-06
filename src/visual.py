import torch
from IOGnet_dp import IOGnet
# from model import IOGnet
from dataset import batch_generator
from matplotlib import pyplot as plt

m = IOGnet()
path = "../model/IOGnet_dr20.json"
checkpoint = torch.load(path, map_location=torch.device('cpu'))
m.load_state_dict(checkpoint['model_state_dict'])
print(checkpoint["mean_loss"])
print(checkpoint["iou"])
print(checkpoint["pixel_acc"])
print(checkpoint["dice_coeff"])

m.eval()
# print(checkpoint["mean_loss"])
g = batch_generator(1, 16,False, False)
print("batch_n = ", next(g))

_, axs = plt.subplots(5,8)

for i in range(5):
    for j in range(8):
        axs[i,j].axis("off")

axs[0, 0].set_title('X')
axs[0, 1].set_title('bbox')
axs[0, 2].set_title('clicks')
axs[0, 3].set_title('y')
axs[0, 4].set_title('pred_y')
axs[0, 5].set_title('th 0.4')
axs[0, 6].set_title('th 0.5')
axs[0, 7].set_title('th 0.6')



for i in range(5):

    X, y, _ = next(g)
    # print(np.array(X).shape)
    with torch.no_grad():
        pred_y = m(X)
    
    bbox = X[0][3].squeeze()
    clicks = X[0][4].squeeze()
    X = X[0][:3].permute(1,2,0)
    y = y[0].squeeze()
    pred_y_th_5 = (pred_y[0] > 0.4).float().squeeze()
    pred_y_th_6 = (pred_y[0] > 0.5).float().squeeze()
    pred_y_th_7 = (pred_y[0] > 0.6).float().squeeze()
    pred_y = pred_y[0].squeeze()

    print(i)
    axs[i, 0].imshow(X)
    axs[i, 1].imshow(bbox)
    axs[i, 2].imshow(clicks)
    axs[i, 3].imshow(y)
    axs[i, 4].imshow(pred_y)
    axs[i, 5].imshow(pred_y_th_5)
    axs[i, 6].imshow(pred_y_th_6)
    axs[i, 7].imshow(pred_y_th_7)


plt.tight_layout(pad=0.3, h_pad=0.3)
plt.show()


