import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import ImageTk,Image 
from shapely.geometry import Point
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
from model import IOGnet
import matplotlib.pyplot as plt

img_file_name = ""
points = []
points_circles = []
borders = []

# TK initialize ----------
window = tk.Tk(className="GST Interactive Segmentation")
window.geometry("600x50")
# TK initialize end ------


# MODEL -------
m = IOGnet()
path = "/home/adrian/skola/2sem/knn/proj/IOGnet_final_bn8.json"
checkpoint = torch.load(path, map_location=torch.device('cpu'))
m.load_state_dict(checkpoint['model_state_dict'])
# -------------


def choose_file():
    global img_file_name
    global img_opened 
    global img_canvas
    global img1
    global window

    # dialog window to choose file

    img_file_name = askopenfilename()

    # show image
    try:
        img_opened = Image.open(img_file_name)
    except AttributeError:
        return
    
    # resize if needed
    x = img_opened.width
    y = img_opened.height
    if  x > 1000 or y > 1000:
        resize_ratio = 1000 / max(x, y)
        x = int(x * resize_ratio)
        y = int(y * resize_ratio)
        img_opened = img_opened.resize((x, y), Image.ANTIALIAS)

    img1 = ImageTk.PhotoImage(img_opened)

    # adapt window size to the image size
    window.update()
    if window.winfo_width() < img1.width() or window.winfo_height() < img1.height():
        x = img1.width() if img1.width() > 600 else 600
        window.geometry(str(x) + "x" + str(img1.height()+55))
    # adapt canvas size to the image size
    img_canvas.config(width=img1.width(), height=img1.height())

    # create image
    img_canvas.create_image(img1.width()/2, img1.height()/2, image=img1)


def reset_clicks():
    global img_canvas
    global points_circles
    global points

    for x in points_circles:
        img_canvas.delete(x)

    points = []


def do_segmentation():
    global img_file_name
    global img_opened
    global points
    global borders

    img = img_opened.convert("RGB")
    trans = transforms.ToTensor()
    tensor = trans(img)


    # CREATE CLICKS MAP
    clicks_map = torch.zeros_like(tensor[0])

    for point in points:
        clicks_map[int(point.y), int(point.x)] = 1


    # CREATE BORDER MAP
    border_map = torch.zeros_like(tensor[0])
    x1 = int(borders[0].x)
    y1 = int(borders[0].y)
    x2 = int(borders[1].x)
    y2 = int(borders[1].y)

    # top
    border_map[y1, x1:x2+1:] = 1
    # bottom
    border_map[y2, x1:x2+1:] = 1
    # left
    border_map[y1:y2+1:, x1] = 1
    # right
    border_map[y1:y2+1, x2] = 1
    
    input = torch.unsqueeze(torch.vstack((tensor, clicks_map[None, :, : ], border_map[None, :, : ])), 0)
    
    
    # CROP input by BBox
    pow2 = [pow(2, x) for x in range(4, 15)]

    x_range_old = x2 - x1
    y_range_old = y2 - y1
    x_range_new = 0
    y_range_new = 0

    if x_range_old not in pow2:
        tmp = np.searchsorted(pow2,[x_range_old,],side='right')[0]
        x_range_new = pow2[tmp] - x_range_old

    if y_range_old not in pow2:
        tmp = np.searchsorted(pow2,[y_range_old,],side='right')[0]
        y_range_new = pow2[tmp] - y_range_old

    # Add padding to get pow2 shape
    # X1
    if x1 - x_range_new / 2 < 0:
        x_range_new -= x1
        x1 = 0
    else:
        x1 -= x_range_new / 2
        x_range_new /= 2

    # X2
    x2 += x_range_new

    # Y1
    if y1 - y_range_new / 2 < 0:
        y_range_new -= y1
        y1 = 0
    else:
        y1 -= y_range_new / 2
        y_range_new /= 2

    # X2
    y2 += y_range_new
    # -----------------------

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    # reshape img tensor - add black padding
    src_shape = input.shape
    target = torch.zeros(src_shape[0], src_shape[1], 2048, 2048)
    
    target[:, :, :src_shape[2], :src_shape[3]] = input

    input = target[:, :, y1:y2, x1:x2]

    # DO A SEGMENTATION ----
    with torch.no_grad():
        pred_y = m(input)

    # Threshold
    pred_y = (pred_y>0.5).float()
    img_plot = plt.imshow(pred_y[0,0,:,:])
    plt.show()
    # -------------------------


    # create mask
    mask = torch.zeros_like(tensor[0])

    # create 3 channel (RGB) tensor of segmentation output
    pred_y = torch.squeeze(pred_y)
    # pred_y = torch.stack((pred_y, pred_y, pred_y))
    print(mask.shape, pred_y.shape)

    mask[y1:y2, x1:x2] = pred_y
    img_plot = plt.imshow(mask)
    #plt.show()

    mask = mask + 1
    mask = mask / 2

    # create 3 channel (RGB) mask
    mask = torch.stack((mask, mask, mask))

    res_image = tensor * mask

    print(res_image)


    # FINAL RESULT IMAGE
    trans = transforms.ToPILImage()
    image = trans(res_image)

    img_plot = plt.imshow(image)
    plt.show()

    # image = ImageTk.PhotoImage(trans)

    # img_canvas.delete("all")
    # img_canvas.create_image(img1.width()/2, img1.height()/2, image=image)
    print("hura")

    pass


def add_click(event):
    global img_canvas
    global points
    global points_circles

    points_circles.append(img_canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill='blue', outline='red', width=2))
    points.append(Point(event.x, event.y))


def add_border(event):
    global img_canvas
    global borders
    global borders_rectangle

    if len(borders) < 2:
        borders.append(Point(event.x, event.y))
    else:
        borders = []
        borders.append(Point(event.x, event.y))

    if len(borders) == 2:
        try:
            img_canvas.delete(borders_rectangle)
        except NameError:
            pass

        borders_rectangle = img_canvas.create_rectangle(borders[0].x, borders[0].y, borders[1].x, borders[1].y, outline='blue', width=2)


def motion(event):
    global img_canvas
    global borders
    global borders_rectangle

    if len(borders) == 1:
        try:
            img_canvas.delete(borders_rectangle)
        except NameError:
            pass

        borders_rectangle = img_canvas.create_rectangle(borders[0].x, borders[0].y, event.x, event.y, outline='blue', width=2)



# BUTTONS --------------
pick_image_btn = tk.Button(window, text="Choose file", command=choose_file)
pick_image_btn.grid(row=0, column=0, sticky="wens")

reset_btn = tk.Button(window, text="Reset clicks", command=reset_clicks)
reset_btn.grid(row=0, column=1, sticky="wens")

start_seg_btn = tk.Button(window, text="Segment", command=do_segmentation)
start_seg_btn.grid(row=0, column=2, sticky="wens")
# BUTTONS end ----------

# CANVAS
img_canvas = tk.Canvas(width=10, height=10, bg='black')
img_canvas.grid(row=1, column=0, columnspan=3)

# MOUSE BUTTONS BIND ------
img_canvas.bind("<Button-1>", add_click)
img_canvas.bind("<Button-3>", add_border)
img_canvas.bind("<Motion>", motion)



# SET EXPANDING SIZE TO BUTTONS DUE TO WINDOW SIZE
for x in range(3):
    tk.Grid.columnconfigure(window, x, weight=1)

tk.Grid.rowconfigure(window, 0, minsize=50)
# -------

window.mainloop()
