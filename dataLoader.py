#!/usr/bin/env python
import xml.etree.ElementTree as ET
import xmlschema
import json
import random
import cv2 

from os import listdir
from os.path import isfile, join

class DataLoader:
    def __init__(self):
        self.config()
        self.train_x = self.getDirectoryContent(self.getConfigVal("trainData"))
        self.train_y = json.load(open(self.getConfigVal("trainLabel")))

    def getDirectoryContent(self, dirName):
        return [dirName + f for f in listdir(dirName) if isfile(join(dirName, f))]

    def getFilenameAnnotation(self, filename):
        return self.train_y[(filename.lstrip("0"))[:-4]]
        

    def config(self, configFile='config.xml', validatorFile='config.xsd'):
        # uncomment for final product
        #xmlschema.validate(configFile, validatorFile)

        xmlTree = ET.parse(configFile)
        self.configRoot = xmlTree.getroot()

    def getConfigVal(self, valName):
        return self.configRoot.find(valName).text

    def getConfigMultipleVals(self, valName):
        vals = []
        for item in self.configRoot.findall(valName):
            vals.append(item.text)
        return vals

    def batchProvider(self, batchSize=2):
        for i in range(0, len(self.train_x), batchSize):
            return [(cv2.imread(img),self.getFilenameAnnotation(img.rsplit('/',1)[-1])) for img in self.train_x[i:i+batchSize]]
