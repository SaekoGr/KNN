from PIL import ImageTk, Image
import torch
import json
import random
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt


def generate_clicks(siluet, bbox):
    clicks_num = 5
    clicks_points = []
    click_map = torch.zeros_like(siluet)
    other_clicks = torch.zeros(4, siluet.shape[1], siluet.shape[2])
    # generate clicks with MonteCarlo method
    while clicks_num > 0:
        valid_point = True
        random_x = random.randint(int(bbox[0]), int(bbox[2])-1)
        random_y = random.randint(int(bbox[1]), int(bbox[3])-1)

        if siluet[0][random_y][random_x] == 1:
            # check distance between points - at least 10px
            for point in clicks_points:
                if abs(point.x - random_x) < 10 and abs(point.y - random_y) < 10:
                    valid_point = False
                    break
            if valid_point:  
                clicks_points.append(Point(random_x, random_y))  
                clicks_num -= 1
    
    # first click
    click_map[0][int(clicks_points[0].y)][int(clicks_points[0].x)] = 1
    del clicks_points[0]

    # other clicks
    for i, point in enumerate(clicks_points):
        other_clicks[i][int(point.y)][int(point.x)] = 1

    return click_map, other_clicks


def generate_b_map(siluet, bbox):
    border_map = torch.zeros_like(siluet)
    
    bx1, by1, bx2, by2 = bbox

    # add noise
    n1, n2, n3, n4 = [random.randint(0, 20) for x in range(4)]
    x1 = int(bx1 - n1 if bx1 - n1 > 0 else 0)
    y1 = int(by1 - n2 if by1 - n2 > 0 else 0)
    x2 = int(bx2 + n3 if bx2 + n3 < siluet.shape[2] else siluet.shape[2]-1)
    y2 = int(by2 + n4 if by2 + n4 < siluet.shape[1] else siluet.shape[1]-1) 

    # top
    border_map[0, y1, x1:x2+1:] = 1
    # bottom
    border_map[0, y2, x1:x2+1:] = 1
    # left
    border_map[0, y1:y2+1:, x1] = 1
    # right
    border_map[0, y1:y2+1, x2] = 1

    return border_map


def get_maps(x_batch, y_batch, bboxes):
    other_clicks_maps = []
    click_maps = []
    b_maps = []

    for i, siluet in enumerate(y_batch):
        # CLICKS
        first_click, other_clicks = generate_clicks(siluet, bboxes[i])

        click_maps.append(first_click)
        other_clicks_maps.append(other_clicks)

        # BORDERS
        b_maps.append(generate_b_map(siluet, bboxes[i]))

    click_maps = torch.stack(click_maps)
    b_maps = torch.stack(b_maps)

    return torch.hstack((x_batch, b_maps, click_maps)), torch.stack(other_clicks_maps)
