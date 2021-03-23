#!/usr/bin/env python
import xml.etree.ElementTree as ET
import xmlschema

class DataLoader:
    def __init__(self):
        pass

    def config(self, configFile='config.xml', validatorFile='config.xsd'):
        # uncomment for final product
        xmlschema.validate(configFile, validatorFile)

        xmlTree = ET.parse(configFile)
        self.configRoot = xmlTree.getroot()


    def getConfigVal(self, valName):
        return self.configRoot.find(valName).text

    def laodDataset(self):
        pass

    def batchProvider(self):
        pass
