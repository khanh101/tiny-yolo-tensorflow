#!/usr/bin/env python3
import sh
import os
import random
import numpy as np
import cv2

data_path = "./data"

images_path = os.path.join(data_path, "images")
labels_path = os.path.join(data_path, "labels")

images_list = os.listdir(images_path)

def create(input_size, flip=1, crop=0.9, angle=10, color = 0.05):
    image_name = random.choice(images_list)
    image_path = os.path.join(images_path, image_name)
    label_name = image_name.split(".")[0] + ".txt"
    label_path = os.path.join(labels_path, label_name)

    #image
    im = cv2.imread(image_path).astype(np.float)
    h, w, _ = im.shape
        
        #rotate
    rot = random.uniform(-angle, +angle)
    M = cv2.getRotationMatrix2D((w/2, h/2), rot, 1)
    im = cv2.warpAffine(im, M, (w, h))
    
        #crop
    size = int(min(w, h) * random.uniform(crop, 1))
    x_min = int(random.uniform(0, w - size))
    y_min = int(random.uniform(0, h - size))
    x_max = x_min + size
    y_max = y_min + size
    im = im[y_min:y_max, x_min:x_max, :]

        #flip
    fl = random.random() < 0.5
    if fl:
        im = cv2.flip(im, 1)
    
       #color
    red = random.uniform(1-color, 1+color)
    blu = random.uniform(1-color, 1+color)
    gre = random.uniform(1-color, 1+color)

    col = np.array([blu, gre, red])
    im = im*col
    im[im<0] = 0
    im[im>255] = 255
        #resize to inputsize
    image = cv2.resize(im, (input_size, input_size), interpolation = cv2.INTER_CUBIC)
    image = image.reshape((1, input_size, input_size, 3))

    #label

    label = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            labeltxt = f.read()
        for objtxt in labeltxt.split("\n"):
            if objtxt == "": continue
            cls, x0, y0, w0, h0, _ = objtxt.split(" ")
            cls = int(cls)
            x0   = float(x0)
            y0   = float(y0)
            w0   = float(w0)
            h0   = float(h0)
            #convert back
            
                #rotate
            rot = np.deg2rad(rot)
            M = np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])
            x0, y0 = 0.5+np.matmul(M, np.array([x0-0.5, y0-0.5]))
                #w0 h0 remain
            
                #crop
            if x0 < x_min/w or x0 > x_max/w or y0 < y_min/h or y0 > y_max/h: continue
            x0 = (x0*w - x_min)/size
            y0 = (y0*h - y_min)/size
            w0 = w0*w/size
            h0 = h0*h/size

                #flip
            if fl:
                x0 = 1-x0
            
            label.append((cls, x0, y0, w0, h0))
    return image, label

def IoU(box1, box2):
    w1, h1 = box1
    w2, h2 = box2
    iou = min(w1, w2) * min(h1, h2)
    return iou

def which_anchor(box):
    anchor = ((10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319))
    dist = []
    for i in range(6):
        dist.append(IoU(anchor[i], box))
    i = dist.index(max(dist))
    return i

def create_array(input_size):
    image, label = create(input_size)
    _, height, width, depth = image.shape
    classes = 80
    out_height = height//32
    out_width = width//32
    out_depth = 3*(5+classes)
    
    X = image
    Y1 = np.random.random((1, out_height, out_width, out_depth))
    Y2 = np.random.random((1, 2*out_height, 2*out_width, out_depth))
    for i in range(3):
        Y1[:, :, :, i*(out_depth//3)] = 1
        Y2[:, :, :, i*(out_depth//3)] = 1
    #convert label to array
    for obj in label:
        cls, x0, y0, w0, h0 = obj
        if x0<0 or x0>=1 or y0<0 or y0>=1: continue
        box = (w0, h0)
        i = which_anchor(box)
        if (i<3): #anchor1
            x = int(out_width*x0)
            y = int(out_height*y0)
            Y1[0, y, x, 0+i*(out_depth//3)] = 1
            Y1[0, y, x, 1+i*(out_depth//3)] = x0
            Y1[0, y, x, 2+i*(out_depth//3)] = y0
            Y1[0, y, x, 3+i*(out_depth//3)] = w0
            Y1[0, y, x, 4+i*(out_depth//3)] = h0
            Y1[0, y, x, 4:(i+1)*(out_depth//3)] = 0
            Y1[0, y, x, cls] = 1
        else: #anchor2
            i = i - 3
            x = int(2*out_width*x0)
            y = int(2*out_height*y0)
            Y2[0, y, x, 0+i*(2*out_depth//3)] = 1 
            Y2[0, y, x, 1+i*(2*out_depth//3)] = x0
            Y2[0, y, x, 2+i*(2*out_depth//3)] = y0
            Y2[0, y, x, 3+i*(2*out_depth//3)] = w0
            Y2[0, y, x, 4+i*(2*out_depth//3)] = h0
            Y2[0, y, x, 4:(i+1)*(2*out_depth//3)] = 0
            Y2[0, y, x, cls] = 1
    return X, Y1, Y2

def create_many_arrays(batch_size, input_size):
    X = []
    Y1 = []
    Y2 = []
    for i in range(batch_size):
        x, y1, y2 = create_array(input_size)
        X.append(x)
        Y1.append(y1)
        Y2.append(y2)
    X = np.vstack(X)
    Y1 = np.vstack(Y1)
    Y2 = np.vstack(Y2)            
    return X, Y1, Y2

def shuffle(batch_size, input_size):
    step = 0
    while (1):
        if (step == 0):
            yield step, None, None, None
        else:
            yield step, X, Y1, Y2
            del X
            del Y1
            del Y2
        step += 1
        X, Y1, Y2 = create_many_arrays(batch_size, input_size)

if __name__ == "__main__":
    image, label = create(416)
    image = image.astype(np.int32).reshape(416,416,3)
    print(image.shape)
    for obj in label:
        cls, x0, y0, w0, h0 = obj
        x1 = int((x0 - w0/2)*416)
        x2 = int((x0 + w0/2)*416)
        y1 = int((y0 - h0/2)*416)
        y2 = int((y0 + h0/2)*416)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,0),2)

    cv2.imwrite("temp.jpg", image)
    sh.eog("temp.jpg")
    sh.rm("temp.jpg")                                            
