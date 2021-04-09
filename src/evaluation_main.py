#!/usr/bin/env python
import torch
from model import IOGnet
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
        comparisom = torch.where((y == prediction) == True, 1, 0).sum().item()
        full_shape = (np.prod([prediction.shape])).item()
        #print(comparisom, full_shape)

        self.pixel_acc_err += comparisom
        self.pixel_total_sum += full_shape

    def intersectionOverUnion(self, y, prediction):
        intersectionArea = (torch.logical_and(y ,prediction).sum()).item()
        unionArea =  (torch.logical_or(y, prediction).sum()).item()
        #print(intersectionArea, unionArea)

        self.unionArea.append(unionArea)
        self.intersectionArea.append(intersectionArea)

    def calculateIoU(self):
        self.iou = np.sum(self.intersectionArea) / np.sum(self.unionArea) * 100


    def calculateDiceLoss(self):
        self.dice_coeff = np.sum(2 * self.intersectionArea) / (np.sum(self.unionArea + self.intersectionArea)) * 100

    def calculatePixelAccuracy(self):
        if(self.pixel_total_sum > 0):
            self.pixel_acc = ((self.pixel_acc_err) / self.pixel_total_sum * 100) 
        else:
            self.pixel_acc = 0


    def getEvaluation(self, epoch_id, output_result=True):
        self.calculatePixelAccuracy()
        self.calculateIoU()
        self.calculateDiceLoss()
        
        if(output_result):
            print("\n================================")
            print("Evaluating model for epoch " + str(epoch_id))
            print("Pixel accuracy " + str(round(self.pixel_acc, 2)) + "%")
            print("Intersection over union " + str(round(self.iou, 2)) + "%")
            print("Dice coefficient " + str(round(self.dice_coeff, 2)) + "%")

def evaluate(epoch_id, model, batch_size=4, min_res_size=16):
    gen = batch_generator(batch_size, min_res_size, False)
    batch_n = next(gen)

    evalMetrics = EvaluationMetrics(batch_n)

    model.zero_grad()

    #s = perf_counter()
    for n in range(batch_n):
        X, y, _, bboxes = next(gen)

        with torch.no_grad():
            prediction = model(X)
        
        # evaluate this batch
        #print(y, prediction)
        evalMetrics.evaluateBatch(y, prediction)

        del X, y, prediction, bboxes

        gc.collect()
        torch.cuda.empty_cache()
        loading(n+1, batch_n)


    #print("total time = ", perf_counter() - s)
    evalMetrics.getEvaluation(epoch_id)

    return evalMetrics.pixel_acc, evalMetrics.iou, evalMetrics.dice_coeff
