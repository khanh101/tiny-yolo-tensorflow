#!/usr/bin/env python3
'''
This script will randomize weights and train the tiny yolo from scratch.
'''
from data.datahandler import shuffle
import tensorflow as tf
import numpy as np
import os
import sys
import shutil
import time

saver = tf.train.import_meta_graph("./graph/tiny-yolo.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess, "./graph/tiny-yolo.ckpt")
    g = sess.graph
    with g.name_scope("TRAINER"):
        X = g.get_tensor_by_name("YOLO/input:0")
        batch_size, height, width, in_channels = X.get_shape().as_list()
        classes = 80
        out_height = height//32
        out_width = width//32
        out_channels = 3*(5+classes)
        h1 = g.get_tensor_by_name("YOLO/output1:0")
        h2 = g.get_tensor_by_name("YOLO/output2:0")
        Y1 = tf.placeholder(shape = (batch_size, out_height, out_width, out_channels), dtype = tf.float32, name = "groundtruth1")
        Y2 = tf.placeholder(shape = (batch_size, 2*out_height, 2*out_width, out_channels), dtype = tf.float32, name = "groundtruth2")
    
        #loss
        h = []
        Y = []

        split_h1 = tf.split(h1, 3, axis = 3)
        for split in split_h1:
            h.append(tf.reshape(split, [batch_size * out_height * out_width, out_channels//3]))

        split_h2 = tf.split(h2, 3, axis = 3)
        for split in split_h2:
            h.append(tf.reshape(split, [batch_size * 2*out_height * 2*out_width, out_channels//3]))

        split_Y1 = tf.split(Y1, 3, axis = 3)
        for split in split_Y1:
            Y.append(tf.reshape(split, [batch_size * out_height * out_width, out_channels//3]))

        split_Y2 = tf.split(Y2, 3, axis = 3)
        for split in split_Y2:                                                                         
            Y.append(tf.reshape(split, [batch_size * 2*out_height * 2*out_width, out_channels//3]))       
    
        h = tf.concat(h, 0)
        Y = tf.concat(Y, 0)
   
        Lcoord = 1
        Lnoobj = 1
        loss_xy = Lcoord*tf.reduce_mean(Y[:,0]*((h[:,1] - Y[:,1])**2 + (h[:,2] - Y[:,2])**2))
        loss_wh = Lcoord*tf.reduce_mean(Y[:,0]*((h[:,3]**0.5 - Y[:,3]**0.5)**2+(h[:,4]**0.5 - Y[:,4]**0.5)**2))
        loss_obj = (-1)*tf.reduce_mean(tf.tile(Y[:,0:1], (1, classes))*(Y[:,5:]*tf.log(h[:,5:]) + (1-Y[:,5:])*tf.log(1-h[:,5:])))
        loss_noobj = (-1*Lnoobj)*tf.reduce_mean(tf.tile(1-Y[:,0:1], (1, classes))*(Y[:,5:]*tf.log(h[:,5:]) + (1-Y[:,5:])*tf.log(1-h[:,5:])))
        loss_p = (-1)*tf.reduce_mean(tf.tile(Y[:,0:1], (1, classes))*tf.log((tf.tile(h[:,0:1], (1, classes)) * Y[:,5:])) + (1-tf.tile(Y[:,0:1], (1, classes)))*tf.log(1-(tf.tile(h[:,0:1], (1, classes)) * Y[:,5:])))

        loss = loss_xy + loss_wh + loss_obj + loss_noobj + loss_p

        optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)
        trainer = optimizer.minimize(loss, name = "trainer")

    if os.path.exists("./train_graph"):
            shutil.rmtree("./train_graph")
    os.mkdir("./train_graph")

    train_writer = tf.summary.FileWriter("./train_graph", g)
    saver = tf.train.Saver()
    tf.summary.histogram("loss", loss)
    merge = tf.summary.merge_all()



    hm_steps = 400000
    sess.run(tf.global_variables_initializer())

    input_size = height

    for batch in shuffle(batch_size, input_size):
        step, Xp, Y1p, Y2p = batch
        if step == 0:
            time.sleep(1)
            continue
        debugger = tf.logical_or(tf.is_nan(loss), tf.is_inf(loss))
#        import pdb
#        pdb.set_trace()
        while (1):
            d, l = sess.run([debugger, loss], feed_dict = {X:Xp, Y1:Y1p, Y2:Y2p})
            if (not d):
                break
            else:
                print("Re-random variables!")
                sess.run(tf.global_variables_initializer())
        summary, _ , lossp, lxy, lwh, lobj, lnoobj, lp = sess.run([merge, trainer, loss, loss_xy, loss_wh, loss_obj, loss_noobj, loss_p], feed_dict = {X: Xp, Y1: Y1p, Y2:Y2p})

        print("Step {} : loss {}".format(step, lossp))
        print("     loss_xy     = {}".format(lxy))
        print("     loss_wh     = {}".format(lwh))
        print("     loss_obj    = {}".format(lobj))
        print("     loss_noobj  = {}".format(lnoobj))
        print("     loss_p      = {}".format(lp))


        train_writer.add_summary(summary, step)
        print("Step {} : loss {}".format(step, lossp))

        if (step % 1250 ==0):
            saver.save(sess, "./train_graph/tiny-yolo-{}.ckpt".format(step))

        



