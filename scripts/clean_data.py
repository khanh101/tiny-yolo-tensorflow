#!/usr/bin/env python3
import os
#import sh
data_path = "../data"

images_path = os.path.join(data_path, "images")
labels_path = os.path.join(data_path, "labels")

count = 0
import pdb
for image_name in os.listdir(images_path):

    image_path = os.path.join(images_path, image_name)
    label_name = image_name.split(".")[0] + ".txt"
    label_path = os.path.join(labels_path, label_name)

    if not os.path.exists(label_path):
        count += 1
        print(count, image_path)
        #sh.touch(label_path)



