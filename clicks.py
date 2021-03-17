from tkinter import Tk, Canvas
from PIL import ImageTk, Image
import json
import random

# Requires annotation file in constructor
class Clicks():
    img_id = ''
    img_name = ''
    annotation_file = ''
    annotations = []
    data = []

    def __init__(self, annotation_file):
        with open(annotation_file) as json_file:
            self.data = json.load(json_file)


    def set_img(self, img_id):
        # TODO - opravit "part1" na skutocnu cestu k obrazkom
        self.img_name = '../part1/' + self.data[img_id]['file_name']
        self.annotations = self.data[img_id]['annotations']


    def generate_clicks(self, segment_index):
        pass



    def show_all(self):
        # Create root
        root = Tk()

        # Create a canvas
        canvas = Canvas(width=640, height=360)
        canvas.pack()

        # Load the image file
        im = Image.open(self.img_name)
        # Put the image into a canvas compatible class, and stick in an
        # arbitrary variable to the garbage collector doesn't destroy it
        canvas.image = ImageTk.PhotoImage(im)
        # Add the image to the canvas, and set the anchor to the top left / north west corner
        canvas.create_image(0, 0, image=canvas.image, anchor='nw')

        for annotation in self.annotations:


            red = ("%02x"%random.randint(0,255))
            green = ("%02x"%random.randint(0,255))
            blue = ("%02x"%random.randint(0,255))

            rand_color= '#'+red+green+blue

            canvas.create_polygon(annotation['segmentation'], fill=rand_color, stipple="gray50", outline='#ffffff')

        root.mainloop()


    def show(self, segmentation):
        # Create root
        root = Tk()

        # Create a canvas
        canvas = Canvas(width=640, height=360)
        canvas.pack()

        # Load the image file
        im = Image.open(self.img_name)
        # Put the image into a canvas compatible class, and stick in an
        # arbitrary variable to the garbage collector doesn't destroy it
        canvas.image = ImageTk.PhotoImage(im)
        # Add the image to the canvas, and set the anchor to the top left / north west corner
        canvas.create_image(0, 0, image=canvas.image, anchor='nw')

        # Show segmentation polygon
        canvas.create_polygon(segmentation, fill='#0000ff', stipple="gray50", outline='')
        # Show segmentation borders
        xs = segmentation[::2]
        ys = segmentation[1::2]
        canvas.create_rectangle(min(xs), min(ys), max(xs), max(ys), width='3', outline='#0000ff')

        root.mainloop()



test_clicks = Clicks('../part1/test_anotation.json')
test_clicks.set_img('391895')
test_clicks.show([20, 20, 300, 20, 300, 300])

