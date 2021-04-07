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
        self.unionArea = []
        self.intersectionArea = []
        self.dice_coeff = []


    def evaluateBatch(self, groundTruth, prediction):
        self.pixelAccuracy(groundTruth, prediction)
        self.intersectionOverUnion(groundTruth, prediction)

    def pixelAccuracy(self, y, prediction):
        ones_count = ((np.abs(y - prediction)).sum()).item()
        full_shape = (np.prod([prediction.shape])).item()

        self.pixel_acc_err += ones_count
        self.pixel_total_sum += full_shape

    def intersectionOverUnion(self, y, prediction):
        intersectionArea = (np.logical_and(y ,prediction).sum()).item()
        unionArea =  (np.logical_or(y, prediction).sum()).item()

        self.unionArea.append(unionArea)
        self.intersectionArea.append(intersectionArea)

    def getIoU(self):
        self.iou = np.sum(self.intersectionArea) / np.sum(self.unionArea) * 100
        return self.iou

    def getDiceLoss(self):
        self.dice_coeff = np.sum(2 * self.intersectionArea) / (np.sum(self.unionArea + self.intersectionArea)) * 100
        return self.dice_coeff

    def getPixelAccuracy(self):
        if(self.pixel_total_sum > 0):
            self.pixel_acc = ((self.pixel_acc_err) / self.pixel_total_sum * 100) 
        else:
            self.pixel_acc = 0
        return self.pixel_acc

    def getEvaluation(self, epoch_id):
        print("\n================================")
        print("Evaluating model for epoch " + str(epoch_id))
        print("Pixel accuracy " + str(round(self.getPixelAccuracy(), 2)) + "%")
        print("Intersection over union " + str(round(self.getIoU(), 2)) + "%")
        print("Dice coefficient " + str(round(self.getDiceLoss(), 2)) + "%")

def evaluate(epoch_id, model, batch_size=4, min_res_size=16):
    threshold=0.5
    gen = batch_generator(batch_size, min_res_size, False)
    batch_n = next(gen)

    evalMetrics = EvaluationMetrics(batch_n)

    model.zero_grad()

    #s = perf_counter()
    for n in range(batch_n):
        X, y, _, bboxes = next(gen)

        with torch.no_grad():
            prediction = model(X)
            prediction = (prediction.cuda()).detach().cpu().clone().numpy()

        # calculate bounding boxes for thresholded prediction
        thresholded_prediction = np.where(prediction >= threshold, 1.0, 0.0)
        
        # evaluate this batch
        evalMetrics.evaluateBatch(y.cpu(), thresholded_prediction)

        del X, y, prediction, bboxes

        gc.collect()
        torch.cuda.empty_cache()
        loading(n+1, batch_n)

    #print("total time = ", perf_counter() - s)
    evalMetrics.getEvaluation(epoch_id)

    return evalMetrics.pixel_acc, evalMetrics.iou, evalMetrics.dice_coeff
