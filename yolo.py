import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *
import cv2


def parse_cfg2(file):
    lines = []
    with open(file, "r") as fin:
        lines = fin.readlines()
    layers = []
    layer = {}
    for l in lines:
        print(l)
        if len(l) <= 2:
            continue
        else:
            if l[0] == '[':
                if len(layer) > 0:
                    layers.append(layer)
                layer = {}
                layer["type"] = l[1:-2]
            elif l[0] == '#':
                continue
            else:
                key, val = l.split("=")
                layer[key.strip()] = val.strip()
    if len(layer) > 0:
        layers.append(layer)
    return layers

def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')     #store the lines in a list
    lines = [x for x in lines if len(x) > 0] #get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']  
    lines = [x.rstrip().lstrip() for x in lines]

    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":               #This marks the start of a new block
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks
#    print('\n\n'.join([repr(x) for x in blocks]))

def create_modules2(blocks):
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
            batch_norm = False
            if "batch_normalize" in layer.keys():
                batch_norm = bool(layer["batch_normalize"])
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
            upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            module.add_module("upsample_%d" % i, upsample)
        elif layer["type"] == "route":
            layer["layers"] = layer["layers"].split(",")
            start = int(layer["layers"][0])
            end = 0
            if len(layer["layers"]) > 1:
                end = int(layer["layers"][1])
            if start > 0:
                start = start - i
            if end > 0:
                end = end - i
            route = EmptyLayer()
            module.add_module("route_%d" % i, route)
            filters = output_filters[i + start]
            if end < 0:
                filters += output_filters[i + end]
            
        elif layer["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_%d" % i, shortcut)
        elif layer["type"] == "yolo":
            mask = list(map(int, layer["mask"].split(",")))
            
            anchors = list(map(int, layer["anchors"].split(",")))
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            
            detection = DetectionLayer(anchors)
            module.add_module("Detection_%d" % i, detection)
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return net_info, module_list

def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing    
    
    module_list = nn.ModuleList()
    
    index = 0    #indexing blocks helps with implementing route  layers (skip connections)

    
    prev_filters = 3
    
    output_filters = []
    
    for x in blocks:
        module = nn.Sequential()
        
        if (x["type"] == "net"):
            continue
        
        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
                
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
                
            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
            
            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
            
            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)
            
            
            
        #If it's an upsampling layer
        #We use Bilinear2dUpsampling
        
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
#            upsample = Upsample(stride)
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)
        
        #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            
            #Start  of a route
            start = int(x["layers"][0])
            
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
                
            
            
            #Positive anotation
            if start > 0: 
                start = start - index
            
            if end > 0:
                end = end - index

            
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            
            
            
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]
                        
            
        
        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            from_ = int(x["from"])
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            
            
        elif x["type"] == "maxpool":
            stride = int(x["stride"])
            size = int(x["size"])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)
            
            module.add_module("maxpool_{}".format(index), maxpool)
        
        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
            
            
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
            
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
        
            
            
        else:
            print("Something I dunno")
            assert False


        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1
        
    
    return (net_info, module_list)




class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
    def forward(self, x, inp_dim, num_classes, confidence):
        x = x.data
        global CUDA
        prediction = x
        prediction = predict_transform(prediction, inp_dim, self.anchors, num_classes, confidence, CUDA)
        return prediction

class Yolo(nn.Module):
    def __init__(self, cfg_file):
        super(Yolo, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward2(self, x, cuda):
        modules = self.blocks[1:]
        outputs = {}
        write = False
        for i, mod in enumerate(modules):
            mod_type = mod["type"]
            if mod_type == "convolutional" or mod_type == "upsample":
                x = self.module_list[i](x)
            elif mod_type == "route":
                layers = mod["layers"]
                layers = list(map(int, layers))
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] -= i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)      # dimensions in PyTroch are B x C x H x W
            elif mod_type == "shortcut":
                from_ = int(mod["from"])
                x = outputs[i-1] + outputs[i+from_]
            elif mod_type == "yolo":
                print("LAYER YOLOY ::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info["height"])
                num_classes = int(mod["classes"])
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, cuda)
                if not write:
                    detections = x
                    write = True
                else:
                    detections = torch.cat((detections, x), 1)
                print("yolo layer detections ", detections.shape)
            outputs[i] = x
            print("end of layer : ", x.shape)
        
        return detections
    
    def forward(self, x, CUDA):
        print("Start yolo::forward")
        detections = []
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer
        
        
        write = 0
        print("len modules = ", len(modules))
        for i in range(len(modules)):        
            
            module_type = (modules[i]["type"])
            if module_type == "convolutional" or module_type == "upsample" or module_type == "maxpool":
                
                x = self.module_list[i](x)
                outputs[i] = x

                
            elif module_type == "route":
                layers = modules[i]["layers"]
                layers = [int(a) for a in layers]
                
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                        
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    
                    
                    x = torch.cat((map1, map2), 1)
                outputs[i] = x
            
            elif  module_type == "shortcut":
                from_ = int(modules[i]["from"])
                x = outputs[i-1] + outputs[i+from_]
                outputs[i] = x
                
            
            
            elif module_type == 'yolo':        
                
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])
                
                #Get the number of classes
                num_classes = int (modules[i]["classes"])
                
                #Output the result
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                
                if type(x) == int:
                    continue

                
                if not write:
                    detections = x
                    write = 1
                
                else:
                    detections = torch.cat((detections, x), 1)
                
                outputs[i] = outputs[i-1]
             
        temp=outputs[0].detach().numpy()
        print("temp size = ", temp.shape)
        print("temp = ", temp[0, :4, :4, :4])
        try:
            return detections
        except:
            return 0
    
    
    def load_weights2(self, file):
        with open(file, "rb") as fin:
            header = np.fromfile(fin, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            weights = np.fromfile(fin, dtype=np.float32)
            ptr = 0
            for i in range(len(self.module_list)):
                mod_type = self.blocks[i+1]["type"]
                if mod_type == "convolutional":
                    layer = self.module_list[i]
                    batch_normalize = False
                    if "batch_normalize" in self.blocks[i+1] and self.blocks[i+1]["batch_normalize"]:
                        batch_normalize = True
                    conv = layer[0]
                    
                    if batch_normalize:
                        bn = layer[1]
                        
                        n_bn_biases = bn.bias.numel()
                        
                        bn_biases = torch.from_numpy(weights[ptr:ptr+n_bn_biases])
                        ptr += n_bn_biases
                        
                        bn_weights = torch.from_numpy(weights[ptr:ptr+n_bn_biases])
                        ptr += n_bn_biases
                        
                        bn_running_mean = torch.from_numpy(weights[ptr:ptr+n_bn_biases])
                        ptr += n_bn_biases
                        
                        bn_running_var = torch.from_numpy(weights[ptr:ptr+n_bn_biases])
                        ptr += n_bn_biases
                        
                        bn_biases = bn_biases.view_as(bn.bias.data)
                        bn_weights = bn_biases.view_as(bn.weight.data)
                        bn_running_mean = bn_biases.view_as(bn.running_mean)
                        bn_running_var = bn_biases.view_as(bn.running_var)
                        
                        bn.bias.data.copy_(bn_biases)
                        bn.weight.data.copy_(bn_weights)
                        bn.running_mean.copy_(bn_running_mean)
                        bn_running_var.copy_(bn_running_var)
                        
                    else:
                        num_biases = conv.bias.numel()
                        
                        conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases])
                        ptr += num_biases
                        
                        conv_biases = conv_biases.view_as(conv.bias.data)
                        
                        conv.bias.data.copy_(conv_biases)
                    
                    num_weights = conv.weight.numel()
                    conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                    ptr += num_weights
                    conv_weights = conv_weights.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_weights)
                    
                    
    def load_weights(self, weightfile):
            #Open the weights file
            fp = open(weightfile, "rb")
        
            #The first 5 values are header information 
            # 1. Major version number
            # 2. Minor Version Number
            # 3. Subversion number 
            # 4,5. Images seen by the network (during training)
            header = np.fromfile(fp, dtype = np.int32, count = 5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]   
            
            weights = np.fromfile(fp, dtype = np.float32)
            
            ptr = 0
            for i in range(len(self.module_list)):
                module_type = self.blocks[i + 1]["type"]
        
                #If module_type is convolutional load weights
                #Otherwise ignore.
                
                if module_type == "convolutional":
                    model = self.module_list[i]
                    try:
                        batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                    except:
                        batch_normalize = 0
                
                    conv = model[0]
                    
                    
                    if (batch_normalize):
                        bn = model[1]
            
                        #Get the number of weights of Batch Norm Layer
                        num_bn_biases = bn.bias.numel()
            
                        #Load the weights
                        bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                        ptr += num_bn_biases
            
                        bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr  += num_bn_biases
            
                        bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr  += num_bn_biases
            
                        bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr  += num_bn_biases
            
                        #Cast the loaded weights into dims of model weights. 
                        bn_biases = bn_biases.view_as(bn.bias.data)
                        bn_weights = bn_weights.view_as(bn.weight.data)
                        bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                        bn_running_var = bn_running_var.view_as(bn.running_var)
            
                        #Copy the data to model
                        bn.bias.data.copy_(bn_biases)
                        bn.weight.data.copy_(bn_weights)
                        bn.running_mean.copy_(bn_running_mean)
                        bn.running_var.copy_(bn_running_var)
                    
                    else:
                        #Number of biases
                        num_biases = conv.bias.numel()
                    
                        #Load the weights
                        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                        ptr = ptr + num_biases
                    
                        #reshape the loaded weights according to the dims of the model weights
                        conv_biases = conv_biases.view_as(conv.bias.data)
                    
                        #Finally copy the data
                        conv.bias.data.copy_(conv_biases)
                        
                    #Let us load the weights for the Convolutional layers
                    num_weights = conv.weight.numel()
                    
                    #Do the same as above for weights
                    conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                    ptr = ptr + num_weights
                    
                    conv_weights = conv_weights.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_weights)

            
        
    
def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)) # BGR -> RGB and HxWxC -> CxHxW
    img_ = img_[np.newaxis, :, :, :]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_, img


#if __name__ == "__main__":
    #cfg = parse_cfg("cfg/yolov3.cfg")
    #print(create_modules(cfg))
    
#    model = Yolo("cfg/yolov3.cfg")    
#    model.load_weights("yolov3.weights")
#    inp, out_img = get_test_input()
#    print("Cuda enabled ? ", torch.cuda.is_available())
#    pred = model(inp, torch.cuda.is_available())
#    print(pred)
#    
#    print(pred.shape)
#    filtered_res = filter_results(pred, 0.4, 80)
#    
#    boxes = filtered_res.cpu().numpy()
#    for b in boxes:
#        pt1 = (int(b[1] - b[3]/2), int(b[2] - b[4]/2))
#        pt2 = (int(b[1] + b[3]/2), int(b[2] + b[4]/2))
#        cv2.rectangle(out_img, pt1, pt2, (0, 255, 0))
#    cv2.imwrite("out.png", out_img)
        
