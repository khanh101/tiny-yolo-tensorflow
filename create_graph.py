#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import sys
import os
import shutil

g1 = tf.Graph()

with g1.as_default() as g:
    with g.name_scope("YOLO"):
        def conv(n, in_name, out_channels, kernel_size, stride, nonlin="relu", keep_prob=1, batchnorm=1):
            in_tensor = g.get_tensor_by_name(in_name)
            batch_size, height, width, in_channels = in_tensor.get_shape().as_list()
            with g.name_scope("conv_{}".format(n)):
                kernel = tf.Variable(tf.random_uniform(shape = [kernel_size, kernel_size, in_channels, out_channels])/ (kernel_size*kernel_size*out_channels) , dtype = tf.float32, name = "kernel")
                scale  = tf.Variable(tf.random_normal(shape = [1, 1, 1, out_channels]), dtype = tf.float32, name = "scales")
                bias   = tf.Variable(tf.random_normal(shape = [1, 1, 1, out_channels]), dtype = tf.float32, name = "biases")
                '''
                conv
                batchnorm + bias + scale
                drop
                nonlin
                '''
                strides = (1, stride, stride, 1)
                conv = tf.nn.conv2d(in_tensor, kernel, strides, padding="SAME", name = "conv")
                if (batchnorm):
                    mean_conv, var_conv = tf.nn.moments(conv, axes = [1,2,3], keep_dims = True)
                    batchnorm = tf.nn.batch_normalization(conv, mean_conv, var_conv, bias, scale, 1e-100, name = "batchnorm")
                else:
                    batchnorm = tf.add(conv, bias, name = "batchnorm")
                drop = tf.nn.dropout(batchnorm, keep_prob, name = "drop")
                if nonlin == "relu":
                    nonlin = tf.nn.leaky_relu(drop)
                elif nonlin == "sigmoid":
                    nonlin = tf.sigmoid(drop)
                elif nonlin == "linear":
                    nonlin = tf.identity(drop)
                else:
                    raise Exception(" \"{}\" is not a nonlinear function!".format(nonlin))
                conv = tf.identity(nonlin, name = "out")
            return conv
 
        def maxpool(n, in_name, kernel_size, stride):
            in_tensor = g.get_tensor_by_name(in_name)
            batch_size, height, width, in_channels = in_tensor.get_shape().as_list()
            with g.name_scope("maxpool_{}".format(n)):
                ksize = [1, kernel_size, kernel_size, 1]
                strides = [1, stride, stride, 1]
                '''
                maxpool
                '''
                maxpool = tf.nn.max_pool(in_tensor, ksize, strides, padding="SAME")
                maxpool = tf.identity(maxpool, name = "out")
            return maxpool
 
        def route(n, n1_name, n2_name):
 
            if (n2_name==None):
                n1 = g.get_tensor_by_name(n1_name)
                route = tf.identity(n1)
            else:
                n1 = g.get_tensor_by_name(n1_name)
                n2 = g.get_tensor_by_name(n2_name)
                route = tf.concat([n1, n2], 3)
            with g.name_scope("route_{}".format(n)):
                route = tf.identity(route, name = "out")
            return route
 
        def upsample(n, in_name, stride):
            in_tensor = g.get_tensor_by_name(in_name)
            batch_size, height, width, in_channels = in_tensor.get_shape().as_list()
            out_channels = in_channels
            with g.name_scope("upsample_{}".format(n)):
                kernel = tf.ones([stride, stride, in_channels, out_channels], name = "kernel")
                output_shape = [batch_size, stride*height, stride*width, in_channels]
                strides = [1, stride, stride, 1]
                padding = "SAME"
                unsample = tf.nn.conv2d_transpose(in_tensor, kernel, output_shape, strides, name = "out")
            return unsample
 
        def yolo(n, in_name, anchor, thresh=0.5):#in tensor has shape (batch_size, height, width, 255)
            in_tensor = g.get_tensor_by_name(in_name)
            batch_size, height, width, in_channels = in_tensor.get_shape().as_list()
            split = tf.split(in_tensor, 3, axis = 3)
            new_split = []
            offset_x_np = np.zeros((batch_size, height, width, in_channels//3))
            for i in range(width):
                offset_x_np[:, :, i, :] = i/width
            offset_y_np = np.zeros((batch_size, height, width, in_channels//3))
            for i in range(height):
                offset_y_np[:, :, i, :] = i/height
            offset_x = tf.constant(offset_x_np, dtype = tf.float32)
            offset_y = tf.constant(offset_y_np, dtype = tf.float32)
            for i in range(3):
                o = split[i][:, :, :, 0:1]
                o = tf.sigmoid(o)
                x = split[i][:, :, :, 1:2]
                x = tf.sigmoid(x)/width + offset_x
                y = split[i][:, :, :, 2:3]
                y = tf.sigmoid(y)/height + offset_y
                wh = split[i][:, :, :, 3:5]
                wh = tf.constant(anchor[i], dtype = tf.float32) * tf.exp(wh)
                c = split[i][:, :, :, 5: ]
                c = tf.sigmoid(c)
                new_split.append(o)
                new_split.append(x)
                new_split.append(y)
                new_split.append(wh)
                new_split.append(c)
                #obj,x,y,w,h,classes
            
            with g.name_scope("yolo_{}".format(n)):
                yolo = tf.concat(split, 3, name = "out")
            return yolo

        height = 416
        width = 416
        anchor1 = ((344,319), (135,169), (81,82))
        anchor2 = ((37,58), (23,27), (10,14))
        classes = 80
        batch_size = 32
        image_depth = 3

        out_height = height//32
        out_width = width//32
        out_depth = 3*(5 + classes)

        X = tf.placeholder(shape = (batch_size, height, width, image_depth), dtype = tf.float32, name = "input")
        #0
        conv_0 = conv(0, "YOLO/input:0", 16, 3, 1)
        #1
        maxpool(1, "YOLO/conv_0/out:0", 2, 2)
        #2
        conv(2, "YOLO/maxpool_1/out:0", 32, 3, 1)
        #3
        maxpool(3, "YOLO/conv_2/out:0", 2, 2)
        #4
        conv(4, "YOLO/maxpool_3/out:0", 64, 3, 1)
        #5
        maxpool(5, "YOLO/conv_4/out:0", 2, 2)
        #6
        conv(6, "YOLO/maxpool_5/out:0", 128, 3, 1)
        #7
        maxpool(7, "YOLO/conv_6/out:0", 2, 2)
        #8
        conv(8, "YOLO/maxpool_7/out:0", 256, 3, 1)
        #9
        maxpool(9, "YOLO/conv_8/out:0", 2, 2)
        #10
        conv(10, "YOLO/maxpool_9/out:0", 512, 3, 1)
        #11
        maxpool(11, "YOLO/conv_10/out:0", 2, 1)
        #12
        conv(12, "YOLO/maxpool_11/out:0", 1024, 3, 1)
        #13
        conv(13, "YOLO/conv_12/out:0", 256, 1, 1)
        #14
        conv(14, "YOLO/conv_13/out:0", 512, 3, 1, keep_prob=0.5)
        #15
        conv(15, "YOLO/conv_14/out:0", 255, 1, 1, nonlin = "linear", batchnorm=0)
        #16
        yolo(16, "YOLO/conv_15/out:0", anchor1)
        #17
        route(17, "YOLO/conv_13/out:0", None)
        #18
        conv(18, "YOLO/route_17/out:0", 128, 1, 1, keep_prob=0.5)
        #19
        upsample(19, "YOLO/conv_18/out:0", 2)
        #20
        route(20, "YOLO/upsample_19/out:0", "YOLO/conv_8/out:0")
        #21
        conv(21, "YOLO/route_20/out:0", 256, 3, 1)
        #22
        conv(22, "YOLO/conv_21/out:0", 255, 1, 1, nonlin = "linear", batchnorm=0)
        #23
        yolo(23, "YOLO/conv_22/out:0", anchor2)

        h1 = tf.identity(g.get_tensor_by_name("YOLO/yolo_16/out:0"), "output1")
        h2 = tf.identity(g.get_tensor_by_name("YOLO/yolo_23/out:0"), "output2")

if os.path.exists("./graph"):
	shutil.rmtree("./graph")
os.mkdir("./graph")

tf.summary.FileWriter("./graph", g)

with tf.Session(graph = g) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, "./graph/tiny-yolo.ckpt")
