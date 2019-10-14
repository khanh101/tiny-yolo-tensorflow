#!/usr/bin/env python3

import argparse
import tensorflow as tf
import numpy as np
import cv2


saver = tf.train.import_meta_graph("./train_graph/tiny-yolo-final.ckpt.meta")

sess = tf.Session()
saver.restore("./train_graph/tiny-yolo-final.ckpt")
g = sess.graph
X = g.get_tensor_by_name("YOLO/input:0")
h = g.get_tensor_by_name("TRAINER/h:0")

scores = h[:,0]
y1 = h[:,2:3] - h[:,4:5]
x1 = h[:,1:2] - h[:,3:4]
y2 = h[:,2:3] + h[:,4:5]
x2 = h[:,1:2] + h[:,3:4]
boxes = tf.concat([y1,x1,y2,x2], axis=1)

prediction = tf.image.non_max_suppression(boxes, scores, 10)


def detect(im):
    Xp = letterbox(im)
    return sess.run(prediction, feed_dict = {X:Xp, "YOLO/dropout:0": 1})

def letterbox(im, size=416):
    h, w, _ = im.shape
    im_out = np.zeros(1, size, size, 3)
    if h>=w:
        new_h, new_w = size, int(size*w/h)
    else:
        new_h, new_w = int(size*h/w), size
    im = cv2.resize(im, (new_w, new_h)).reshape(1, new_h, new_w, 3)
    im_out[:, 0:new_h, 0:new_w, 0:3] = im
    return im_out


def draw():
    pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input image")
    args = vars(ap.parse_args())

    image_path = args["input"]

    im = cv2.imread(image_path)

    print(detect(im))




