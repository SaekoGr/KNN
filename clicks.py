from tkinter import Tk, Canvas
from PIL import ImageTk, Image
import json
import random
from shapely.geometry import Polygon, Point


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


    def generate_clicks(self, segment_index, num_of_clicks):
        segmentation = self.annotations[segment_index]['segmentation']
        
        xs = [x[::2] for x in segmentation]
        ys = [x[1::2] for x in segmentation]
        minx = min([min(x) for x in xs])
        maxx = max([max(x) for x in xs])
        miny = min([min(y) for y in ys])
        maxy = max([max(y) for y in ys])

        points = []
        polys = []
        
        # iterate trought all polygons in current segmentation
        for segment in segmentation:
            polys.append(Polygon(list(zip(segment[::2], segment[1::2]))))
        # poly = Polygon(segmentation[0])
        

        while len(points) < num_of_clicks:
            random_point = Point([random.uniform(minx, maxx), random.uniform(miny, maxy)])
            for poly in polys:
                if (random_point.within(poly)):
                    points.append(random_point)
                    break

        border_min = Point([minx, miny])
        border_max = Point([maxx, maxy])

        return (border_min, border_max), points


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


    def show(self, segmentation, border_clicks, pos_clicks):
        # Create root
        root = Tk()

        # Create a canvas
        canvas = Canvas(width=640, height=360)
        canvas.pack()

        # SHOW IMAGE ----
        # Load the image file
        im = Image.open(self.img_name)
        # Put the image into a canvas compatible class, and stick in an
        # arbitrary variable to the garbage collector doesn't destroy it
        canvas.image = ImageTk.PhotoImage(im)
        # Add the image to the canvas, and set the anchor to the top left / north west corner
        canvas.create_image(0, 0, image=canvas.image, anchor='nw')

        # RENDER CLICKS AND SEGMENTATION ----

        # Show segmentation borders
        canvas.create_rectangle(border_clicks[0].x, border_clicks[0].y, border_clicks[1].x, border_clicks[1].y, width='3', outline='#0000ff')

        # Show positive clicks
        for click in pos_clicks:
            canvas.create_oval(click.x - 3, click.y - 3, click.x + 3, click.y + 3, outline='#000000', fill='#ff0000')

        root.mainloop()



test_clicks = Clicks('../part1/test_anotation.json')
test_clicks.set_img('391895')
border_clicks, pos_clicks = test_clicks.generate_clicks(0, 3)
test_clicks.show([20, 20, 300, 20, 300, 300], border_clicks, pos_clicks)

