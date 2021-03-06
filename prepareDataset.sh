#!/bin/bash

echo "Downloading datasets"

FILE_DATA=train2017.zip
FILE_LABEL=stuff_annotations_trainval2017.zip

DATA_FOLDER=train2017
LABEL_FOLDER=stuff_annotations_trainval2017

DATA_DOWNLOAD=http://images.cocodataset.org/zips/train2017.zip
LABEL_DOWNLAOD=http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

if [ -d "$DATA_FOLDER" ]; then
    echo "$DATA_FOLDER exists, not downloading again"
else
    if [ -f "$FILE_DATA" ]; then
        echo "$FILE_DATA exists, unzipping"
        
    else
        echo "Downloading Data file: $FILE_DATA"
        wget $DATA_DOWNLOAD
        echo "Unzipping..."
    fi
    unzip $FILE_DATA
fi


if [ -d "$LABEL_FOLDER" ]; then
    echo "$LABEL_FOLDER exists, not downloading again"
else
    if [ -f "$FILE_LABEL" ]; then
        echo "$FILE_LABEL exists, unzipping"
        
    else
        echo "Downloading Data file: $FILE_DATA"
        wget $LABEL_DOWNLAOD
        echo "Unzipping..."
    fi
    unzip $FILE_LABEL
fi
