from PIL import ImageTk, Image
import torch
import random
from shapely.geometry import Point
import numpy as np


def check_siluet_borders(siluet, x, y, size_dec_x, size_dec_y):
    valid_pixels_cnt = 0.0

    # LEFT
    if x-size_dec_x > 0 and y-size_dec_y > 0 and y+size_dec_y < siluet.shape[0]:
        valid_pixels_cnt += torch.sum(siluet[y-size_dec_y : y+size_dec_y, x-size_dec_x])

    # RIGHT
    if x-size_dec_x < siluet.shape[1] and y-size_dec_y > 0 and y+size_dec_y < siluet.shape[0]:
        valid_pixels_cnt += torch.sum(siluet[y-size_dec_y : y+size_dec_y, x+size_dec_x])

    # TOP
    if x-size_dec_x > 0 and x+size_dec_x < siluet.shape[1] and y-size_dec_y > 0:
        valid_pixels_cnt += torch.sum(siluet[y-size_dec_y, x-size_dec_x : x+size_dec_x])

    # RIGHT
    if x-size_dec_x > 0 and x+size_dec_x < siluet.shape[1] and y-size_dec_y < siluet.shape[0]:
        valid_pixels_cnt += torch.sum(siluet[y+size_dec_y, x-size_dec_x : x+size_dec_x])

    checked_pixels_cnt = 2*size_dec_x + 2* size_dec_y

    if valid_pixels_cnt / checked_pixels_cnt > 0.95:
        return True

    return False



def generate_clicks(siluet, bbox, other_clicks_num):
    clicks_num = 1+other_clicks_num
    clicks_points = []
    clicks_map = torch.zeros_like(siluet)

    bbox_dec_size_x = int((bbox[2] - bbox[0]) * 0.05) 
    bbox_dec_size_y = int((bbox[3] - bbox[1]) * 0.05) 

    # generate clicks with MonteCarlo method
    for i in range(100):
        random_x = random.randint(int(bbox[0] + bbox_dec_size_x), int(bbox[2] - bbox_dec_size_x)-1)
        random_y = random.randint(int(bbox[1] + bbox_dec_size_y), int(bbox[3] - bbox_dec_size_y)-1)

        if siluet[0][random_y][random_x] == 1:    
            if check_siluet_borders(siluet[0], random_x, random_y, bbox_dec_size_x, bbox_dec_size_y):      
                clicks_points.append(Point(random_x, random_y))  
                clicks_num -= 1
        
        if clicks_num == 0:
            break

    # other clicks
    for i, point in enumerate(clicks_points):
        clicks_map[0][int(point.y)][int(point.x)] = 1

    return clicks_map
  
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
    clicks_num = round(np.random.exponential(0.3)) + 1
    click_maps = []
    b_maps = []

    for siluet, bbox in zip(y_batch, bboxes):
        # CLICKS
        clicks = generate_clicks(siluet, bbox, clicks_num)

        click_maps.append(clicks)

        # BORDERS
        b_maps.append(generate_b_map(siluet, bbox))

    click_maps = torch.stack(click_maps)
    b_maps = torch.stack(b_maps)

    return torch.hstack((x_batch, b_maps, click_maps)), b_maps
