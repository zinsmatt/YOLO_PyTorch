#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:48:30 2019

@author: mzins
"""

import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from yolo import Yolo
import random



def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    
    return parser.parse_args()



def load_classes(file):
    names = []
    with open(file, "r") as fin:
        names = fin.readlines()
    return names[:-1]



#
#args = arg_parse()
#images = args.images
#batch_size = args.batch_size
#confidence = float(args.confidence)
#nms_thresh = float(args.nms_thresh)

images = "dog-cycle-car.png"
batch_size = 1
confidence = 0.5
nms_thresh = 0.4

start = 0
cuda = torch.cuda.is_available()

num_classes = 80
classes = load_classes("data/coco.names")

print("Loading the network.....")
model = Yolo(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

if cuda:
    model.cuda()
    
model.eval()


read_dir = time.time()

try:
    imList = [os.path.join(os.path.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imList = []
    imList.append(os.path.join(os.path.realpath('.'), images))
except FileNotFoundError:
    print("No file or directory with name {}".format(images))
    exit()


if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imList]

im_batches = list(map(prepare_image, loaded_ims, [inp_dim for x in range(len(imList))]))
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
if cuda:
    im_dim_list = im_dim_list.cuda()
    

leftover = 0
if len(im_dim_list) % batch_size:
    leftover = 1
if batch_size != 1:
    num_batches = len(imList) // batch_size + leftover
    im_batches = [torch.cat((im_batches[i*batch_size:min((i+1)*batch_size, len(im_batches))])) for i in range(num_batches)]
    
    
    
write = 0
start_det_loop = time.time()
for i, batch in enumerate(im_batches):
    #load the image 
    start = time.time()
    if cuda:
        batch = batch.cuda()

    prediction = model(Variable(batch, volatile = True), cuda)

    prediction = filter_results(prediction, confidence, num_classes, nms_conf=nms_thresh)

    end = time.time()

    if type(prediction) == int:

        for im_num, image in enumerate(imList[i*batch_size: min((i +  1)*batch_size, len(imList))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imList 

    if not write:                      #If we have't initialised output
        output = prediction  
        write = 1
    else:
        output = torch.cat((output,prediction))

    for im_num, image in enumerate(imList[i*batch_size: min((i +  1)*batch_size, len(imList))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if cuda:
        torch.cuda.synchronize()       

try:
    output
except NameError:
    print ("No detections were made")
    exit()
    
im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long()*0)
scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)


output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
output[:,1:5] /= scaling_factor
for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
draw = time.time()

def write(x, results):
    color = np.random.random(3) * 255
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    print(x)
    img = results[int(x[0]*0)]
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

list(map(lambda x: write(x, loaded_ims), output))
det_names = ["det_{}.png".format(i) for i in range(len(imList))]
list(map(cv2.imwrite, det_names, loaded_ims))
end = time.time()