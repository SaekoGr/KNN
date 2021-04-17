import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import ImageTk,Image 

img_file_name = ""


# TK initialize ----------
window = tk.Tk(className="GST Interactive Segmentation")
window.geometry("600x50")
# TK initialize end ------

def do_clicks():
    global window
    global img_canvas


def choose_file():
    global img_file_name 
    global img_canvas
    global img1
    global window

    # dialog window to choose file

    img_file_name = askopenfilename()

    # show image
    try:
        img1 = ImageTk.PhotoImage(Image.open(img_file_name))
    except AttributeError:
        return

    # adapt window size to the image size
    window.update()
    if window.winfo_width() < img1.width() or window.winfo_height() < img1.height():
        x = img1.width() if img1.width() > 600 else 600
        window.geometry(str(x) + "x" + str(img1.height()+55))
    # adapt canvas size to the image size
    img_canvas.config(width=img1.width(), height=img1.height())

    # create image
    img_canvas.create_image(img1.width()/2, img1.height()/2, image=img1)

    do_clicks()



def reset_clicks():
    pass


def do_segmentation():
    pass



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
canvas.bind("<Button-1>", callback)




for x in range(3):
    tk.Grid.columnconfigure(window, x, weight=1)

tk.Grid.rowconfigure(window, 0, minsize=50)

window.mainloop()
