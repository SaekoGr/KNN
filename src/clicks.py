from PIL import ImageTk, Image
import torch
import json
import random
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt


def generate_clicks(siluet, bbox):
    clicks_num = 5

    click_map = torch.zeros_like(siluet)
    # generate clicks with MonteCarlo method
    while clicks_num > 0:
        random_x = int(random.uniform(bbox[0], bbox[2]))
        random_y = int(random.uniform(bbox[1], bbox[3]))

        if siluet[0][random_y][random_x] == 1:
            click_map[random_y][random_x] = 1
            clicks_num -= 1
    
    return click_map


def generate_b_map(siluet, bbox):
    border_map = torch.zeros_like(siluet)

    x1, y1, x2, y2 = bbox

    # top
    border_map[y1, x1:y2+1:] = 1
    # bottom
    border_map[y2, x1:y2+1:] = 1
    # left
    border_map[y1:y2+1:, x1] = 1
    # right
    border_map[y1:y2+1, x2] = 1

    return border_map


def get_maps(x_batch, y_batch, bboxes):
    click_maps = []
    b_maps = []

    for i, siluet in enumerate(y_batch):
        # CLICKS
        click_maps.append(generate_clicks(siluet, bboxes[i]))

        # BORDERS
        b_maps.append(generate_b_map(siluet, bboxes[i]))

    click_maps = torch.stack(click_maps)
    b_maps = torch.stack(b_maps)

    # TODO opravit spojenie do jedneho tenzoru
    return x_batch.cat((click_maps, b_maps), 1)
