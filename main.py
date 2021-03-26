#!/usr/bin/env python
from dataLoader import DataLoader

### parsing input arguments here

### create/train/load model

dataLoad = DataLoader()
print(dataLoad.batchProvider())
