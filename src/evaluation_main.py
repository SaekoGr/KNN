#!/usr/bin/env python
import torch
from model import PSPnet
from dataset import batch_generator, loading
import numpy as np
import gc
from time import perf_counter

if torch.cuda.is_available():
  dev = "cuda:0" 
else:
  dev = "cpu"  
device = torch.device(dev)

class EvaluationMetrics:
    def __init__(self, batch_n):
        self.batch_n = batch_n
        self.pixel_total_sum = 0
        self.pixel_acc_err = 0
        self.iou = []
        self.dice_coeff = []
        self.dice_loss = []

    def evaluateBatch(self, groundTruth, prediction, groundBbox, predictionBbox):
        #print("Evaluated model by Pixel accuracy")
        self.pixelAccuracy(groundTruth, prediction)
        [self.intersectionOverUnion(groundBbox[i], predictionBbox[i]) for i in range(len(groundBbox))]

    def pixelAccuracy(self, y, prediction):
        ones_count = ((np.abs(y - prediction)).sum()).item()
        full_shape = (np.prod([prediction.shape])).item()

        self.pixel_acc_err += ones_count
        self.pixel_total_sum += full_shape

    def intersectionOverUnion(self, groundBbox, predictionBbox):
        #print("IoU")
        #print("Ground truth " + str(groundBbox))
        #print("Evaluating by IoU")
        area1 = abs(groundBbox[3] - groundBbox[1]) * abs(groundBbox[2] - groundBbox[0])
        #print("Ground truth area " + str(area1))

        #print("Prediction bbox " + str(predictionBbox))
        area2 = abs(predictionBbox[3] - predictionBbox[1]) * abs(predictionBbox[2] - predictionBbox[0])
        #print("Prediction area " + str(area2))

        ### intersection rectangle coordinates
        x5 = max(groundBbox[1], predictionBbox[1])
        y5 = max(groundBbox[0], predictionBbox[0])
        x6 = min(groundBbox[3], predictionBbox[3])
        y6 = min(groundBbox[2], predictionBbox[2])

        intersectionArea = abs(x6 - x5) * abs(y6 - y5)
        #print("Intersection area " + str(intersectionArea))
        unionArea =  area1 + area2 - intersectionArea
        iou = intersectionArea / unionArea
        #print("IoU " + str(iou))

        dice_coefficient = (2*intersectionArea) / (unionArea + intersectionArea)

        #print("Dice coefficient " + str(dice_coefficient))
        self.dice_coeff.append(dice_coefficient.item())
        self.dice_loss.append((1 - dice_coefficient).item())

        self.iou.append(iou.item())

    def getIoU(self):
        #print(self.iou)
        return np.sum(self.iou) / self.batch_n * 100

    def getDiceLoss(self):
        #print(self.dice_loss)
        return np.sum(self.dice_loss) / self.batch_n * 100

    def getPixelAccuracy(self):
        if(self.pixel_total_sum > 0):
            return ((self.pixel_acc_err) / self.pixel_total_sum * 100)
        else:
            return 0

    def getEvaluation(self):
        print("\n================================")
        print("Evaluating model:")
        print("Pixel accuracy " + str(round(self.getPixelAccuracy(), 2)) + "%")
        print("Average intersection over union " + str(round(self.getIoU(), 2)) + "%")
        print("Average Dice Loss " + str(round(self.getDiceLoss(), 2)) + "%")


def bbox(points):
    """
    Calculates bounding box by finding the first nonzero row/column

    return rectangle given by 2 points: (y_min, x_min, y_max, x_max)
    """
    x_range, y_range = points.shape

    #print(points.sum())

    x_min = 0
    x_max = x_range - 1
    y_min = 0
    y_max = y_range - 1

    # find x_min
    #print("XMIN")
    for i in range(x_range):
        #print(np.sum(points[i,:]))
        if np.sum(points[i,:]) > 0.0:
            x_min = i
            break

    # find y_min
    #print("YMIN")
    for i in range(y_range):
        #print(np.sum(points[:,i]))
        if np.sum(points[:,i]) > 0.0:
            y_min = i
            break

    ### max needs one correction
    # find x_max
    #print("XMAX")
    for i in range(x_range - 1, -1, -1):
        #print(np.sum(points[i,:]))
        if np.sum(points[i,:]) > 0.0:
            x_max = i
            break

    #print("YMAX")
    for i in range(y_range - 1, -1, -1):
        #print(np.sum(points[:,i]))
        if np.sum(points[:,i]) > 0.0:
            y_max = i
            break

    return (y_min, x_min, y_max, x_max)

if __name__ == "__main__":
    batch_size = 4
    threshold = 0.5

    gen = batch_generator(batch_size, 16, False)
    model = PSPnet() # this will get loaded: load_model()
    model.to(device)

    batch_n = next(gen)

    evalMetrics = EvaluationMetrics(batch_n)

    model.zero_grad()

    s = perf_counter()
    for n in range(batch_n):
        X, y, bboxes = next(gen)
        #print(X.shape, y.shape, bboxes)

        with torch.no_grad():
            prediction = model(X)
            prediction = (prediction.cuda()).detach().cpu().clone().numpy()

        #print(prediction.shape)
        thresholded_prediction = np.where(prediction >= threshold, 1.0, 0.0)
        # calculate bounding boxes for thresholded prediction
        prediction_bboxes = [bbox(thresholded_prediction[i][0]) for i in range(batch_size)]

        # evaluate this batch
        #print("Evaluating")
        #print(y.shape, prediction.shape)
        evalMetrics.evaluateBatch(y.cpu(), thresholded_prediction, bboxes, prediction_bboxes)

        del X, y, bboxes, prediction

        gc.collect()
        torch.cuda.empty_cache()
        loading(n+1, batch_n)
        # uncomment for final product
        #break

    print("total time = ", perf_counter() - s)
    evalMetrics.getEvaluation()

    print("\nEvaluation completed")

