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
transI = transforms.ToPILImage()

# TK initialize ----------
window = tk.Tk(className="GST Interactive Segmentation")
window.geometry("600x50")
# TK initialize end ------


# MODEL -------
m = IOGnet()
path = "/home/adrian/skola/2sem/knn/proj/IOGnet_final_bn8.json"
# path = "../model/IOGnet_final_bn7.json"
checkpoint = torch.load(path, map_location=torch.device('cpu'))
m.load_state_dict(checkpoint['model_state_dict'])
m = m.eval()
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
    global points_circles
    global borders_rectangle

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

    if x1 > x2:
        x1, x2 = x2, x1
    
    if y1 > y2:
        y1, y2 = y2, y1

    # top
    border_map[y1, x1:x2+1:] = 1
    # bottom
    border_map[y2, x1:x2+1:] = 1
    # left
    border_map[y1:y2+1:, x1] = 1
    # right
    border_map[y1:y2+1, x2] = 1
    
    x_pad = (16 - ((x2-x1) % 16)) / 2
    y_pad = (16 - ((y2-y1) % 16)) / 2

    # FIXME možný problém, pokud by už nebyli pixely na krajích
    x1 -= int(np.floor(x_pad))
    x2 += int(np.ceil(x_pad))
    y1 -= int(np.floor(y_pad))
    y2 += int(np.ceil(y_pad))


    input = torch.unsqueeze(torch.vstack((tensor, border_map[None, :, : ], clicks_map[None, :, : ])), 0)
    input = input[:,:,y1:y2,x1:x2]

    with torch.no_grad():
        y_pred = m(input)
    # Threshold
    y_pred = (y_pred>0.5).float()

    # create mask
    mask = torch.zeros_like(tensor[0])

    # create 3 channel (RGB) tensor of segmentation output
    pred_y = torch.squeeze(y_pred)

    mask[y1:y2, x1:x2] = pred_y

    # red_mask = torch.where(mask == 0, 0.3, 1.0)
    mask = torch.where(mask == 0, 0.3, 1.0)

    # create 3 channel (RGB) mask
    mask = torch.stack((mask, mask, mask))
    res_image = tensor * mask

    # FINAL RESULT IMAGE
    image = ImageTk.PhotoImage(image=transI(res_image))

    # CREATE FINAL IMAGE
    img_canvas.create_image(0,0, anchor="nw", image=image)

    # redraw borders
    img_canvas.tag_raise(borders_rectangle)
    # redraw points
    for point in points_circles:
        img_canvas.tag_raise(point)


    window.mainloop()



def add_click(event):
    global img_canvas
    global points
    global points_circles

    points_circles.append(img_canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill='blue', outline='red', width=1))
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
