from tkinter import Tk, Canvas
from PIL import ImageTk, Image
import torch
import json
import random
from shapely.geometry import Polygon, Point

#######################################################################
# USAGE
#######################################################################
# 1. create instance of class Clicks
# 2. Set image to work with by set_img(data, img_id), where:
#       data - loaded annotation file
#       img_id - id from annotation file
# 3. Use get_maps(segment_index, num_of_clicks), where:
#       segment_index - index in annotation file/annotations/segmentation 
#           - which segment to generate from 
#       num_of_clicks - count of clicks to be generated
#       RETURNS:
#       b_map, c_map - border map, clicks map
#######################################################################


class Clicks():
    img_name = ''
    img_width = 0
    img_height = 0
    annotations = []


    def set_img(self, data, img_id):
        # TODO - opravit "part1" na skutocnu cestu k obrazkom
        self.img_name = '../part1/' + data[img_id]['file_name']
        self.img_width = data[img_id]['width']
        self.img_height = data[img_id]['height']
        self.annotations = data[img_id]['annotations']


    # Creates maps from received points
    def create_maps(self, borders, pos_clicks):
        b1x = int(borders[0].x)
        b2x = int(borders[1].x)
        b1y = int(borders[0].y)
        b2y = int(borders[1].y)
        
        # create full borders - not only edges of 1
        borders_map = torch.zeros((self.img_height, self.img_width), dtype=int)
        # top border
        borders_map[b1y, b1x:b2x+1:] = 1
        # bottom border
        borders_map[b2y, b1x:b2x+1:] = 1
        # left border
        borders_map[b1y:b2y+1:, b1x] = 1
        # right border
        borders_map[b1y:b2y+1:, b2x] = 1

        pos_clicks_map = torch.zeros((self.img_height, self.img_width), dtype=int)
        for point in pos_clicks:
            pos_clicks_map[int(point.y)][int(point.x)] = 1
        
        
        return borders_map, pos_clicks_map

    # Needs segment index - from which object to generate
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


    # Generates points and creates maps according to it
    def get_maps(self, segment_index, num_of_clicks):
        # border points, click points
        b_points, c_points = self.generate_clicks(segment_index, num_of_clicks)
        # border map, click map
        b_map, c_map = self.create_maps(b_points, c_points)

        return b_map, c_map


    # Render image and ground truth
    def show_all(self):
        # Create root
        root = Tk()

        # Create a canvas
        canvas = Canvas(width=self.img_width, height=self.img_height)
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


    # Render image, chosen segment and generated clicks
    def show(self, border_clicks, pos_clicks):
        # Create root
        root = Tk()

        # Create a canvas
        canvas = Canvas(width=self.img_width, height=self.img_height)
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


    # Render maps - black "0" pixels, while "1" pixels
    def show_maps(self, b_map, c_map):
        # Create root
        root = Tk()

        # Create a canvas
        canvas = Canvas(width=self.img_width, height=self.img_height, background='#000000')
        canvas.pack()

        for x in range(self.img_width-1):
            for y in range(self.img_height-1):
                if b_map[y][x] == 1 or c_map[y][x] == 1:
                    canvas.create_rectangle(x,y,x,y, fill='#ffffff', outline='')


        root.mainloop()


# ----------------------------------------------------------------
# JUST FOR TESTING -----------------------------------------------
# # loads annotation file
# def open_annotation(annotation_file):
#     with open(annotation_file) as json_file:
#             data = json.load(json_file)

#     return data


# # prepares annotation into local variable
# data = open_annotation('../part1/test_anotation.json')

# # init of class Clicks
# test_clicks = Clicks()
# # sets testing image
# test_clicks.set_img(data, '391895')
# # generates 3 clicks from 0th object in list
# border_clicks, pos_clicks = test_clicks.generate_clicks(0, 3)
# b_map, c_map = test_clicks.get_maps(0, 3)

# x, y = test_clicks.create_maps(border_clicks, pos_clicks)

# test_clicks.show_maps(b_map, c_map)

