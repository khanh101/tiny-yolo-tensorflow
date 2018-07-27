#!/bin/sh
wget -c https://pjreddie.com/media/files/train2014.zip
wget -c https://pjreddie.com/media/files/coco/labels.tgz
unzip train2014.zip -d ./
tar xzf labels.tgz -C ./
mkdir ./data
mv ./labels/train2014 ./data/labels
mv ./train2014 ./data/images
rm -r ./labels
