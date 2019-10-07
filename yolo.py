import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



def parse_cfg(file):
    lines = []
    with open(file, "r") as fin:
        lines = fin.readlines()
    layers = []
    layer = {}
    for l in lines:
        if len(l) <= 2:
            continue
        else:
            if l[0] == '[':
                if len(layer) > 0:
                    layers.append(layer)
                layer = {}
                layer["name"] = l[1:-2]
            elif l[0] == '#':
                continue
            else:
                key, val = l.split("=")
                layer[key.strip()] = val.strip()
    return layers


def create_modules(blocks):
    net_info = blocks[0]
    if net_info["type"] != "net":
        print("Error the first block does not contain the net parameters")
    module_list = nn.ModuleList()
    prev_filters = 3    # nb of filters in the previous layer 
                        # (this is the depth of the current feature map)
    output_filters = [] # keep track of the number of output filter of each block
    
    for i, layer in enumerate(blocks[1:]):
        module = nn.Sequential()
        
        if layer["type"] == "convolutional":
            activation = layer["activation"]
            
            batch_norm = bool(layer["batch_normalize])
            bias = not batch_norm
            filters = int(layer["filters"])
            use_padding = bool(layer["pad"])
            kernel_size = int(layer["size"])
            stride = int(layer["stride"])
            
            pad = 0
            if use_padding:
                pad = (kernel_size - 1) // 2
            
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_%d" % i, conv)
            
            if batch_norm:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_%d" % i, bn)
            
            if activation == "leaky":
                activ = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_%d" % i, activ)
        elif layer["type"] == "upsample":
            stride = int(layer["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsample_%d" % i, upsample)
        elif layer["type"] == "route":
            layer["layers"] = layer["layers"].split(",")
            start = int(layer["layers"][0])
            end = 0
            if len(layer["layers"]) > 1:
                end = int(layer["layers"][1])
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            
                

cfg = parse_cfg("cfg/yolov3.cfg")