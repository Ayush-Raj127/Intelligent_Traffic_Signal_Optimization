'''*** Import Section ***'''
from __future__ import division                     
from collections import Counter                     
import argparse                                     
import os                                           
import os.path as osp                               
import pickle as pkl                                
import pandas as pd                                 
import time                                         
import sys                                          
import torch                                        
from torch.autograd import Variable                 
import cv2                                          
import warnings                                     
warnings.filterwarnings('ignore')                                       

print('\033[1m' + '\033[91m' + "Kickstarting YOLO...\n")
from util.parser import load_classes                
from util.model import Darknet                      
from util.image_processor import preparing_image    
from util.utils import non_max_suppression          
from util.dynamic_signal_switching import switch_signal
from util.dynamic_signal_switching import avg_signal_oc_time

# Create an outputs directory for our visually processed images
if not os.path.exists("outputs"):
    os.makedirs("outputs")

#*** Parsing Arguments to YOLO Model ***
def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO Vehicle Detection Model for Intelligent Traffic Management System')
    # MODIFIED LINE BELOW: changed default="vehicles-on-lanes" to default="inputs"
    parser.add_argument("--images", dest='images', help="Image / Directory containing images", default="inputs", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence_score", dest="confidence", help="Confidence Score", default=0.3)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.3)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="config/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default="weights/yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network.", default="416", type=str)
    return parser.parse_args()

args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

#***Loading Dataset Class File***
classes = load_classes("data/idd.names")

#***Setting up the neural network***
model = Darknet(args.cfgfile)
print('\033[0m' + "Input Data Passed Into YOLO Model..." + u'\N{check mark}')
model.load_weights(args.weightsfile)
print('\033[0m' + "YOLO Neural Network Successfully Loaded..." + u'\N{check mark}')
print('\033[0m')
model.hyperparams["height"] = args.reso
inp_dim = int(model.hyperparams["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32
num_classes = model.num_classes
print('\033[1m' + '\033[92m' + "Performing Vehicle Detection with YOLO Neural Network..." + '\033[0m' + u'\N{check mark}')

if CUDA:
    model.cuda()
model.eval()

#***Vehicle Detection Phase***
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = [osp.join(osp.realpath('.'), images)]
except FileNotFoundError:
    print(f"No Input with the name {images}")
    exit()

loaded_ims = [cv2.imread(x) for x in imlist]
im_batches = list(map(preparing_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

leftover = 0
if (len(im_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    im_batches = [torch.cat((im_batches[i * batch_size:min((i + 1) * batch_size, len(im_batches))])) for i in range(num_batches)]

write = 0
if CUDA:
    im_dim_list = im_dim_list.cuda()

lane_count_list = []
input_image_count = 0
denser_lane = 0
lane_with_higher_count = 0

print('\n\033[1m' + "-"*120)
print('\033[1m' + "SUMMARY")
print('\033[1m' + "-"*120)

for i, batch in enumerate(im_batches):
    vehicle_count = 0
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    with torch.no_grad():
        prediction = model(Variable(batch))

    prediction = non_max_suppression(prediction, confidence, num_classes, nms_conf=nms_thesh)
    end = time.time()

    if type(prediction) == int:
        continue

    prediction[:, 0] += i * batch_size 

    if not write: 
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))

    for im_num, image in enumerate(imlist[i * batch_size:min((i + 1) * batch_size, len(imlist))]):
        vehicle_count = 0
        input_image_count += 1
        im_id = i * batch_size + im_num
        
        # --- NEW CODE: Image processing and bounding box scaling ---
        orig_im = loaded_ims[im_id].copy()
        inp_dim_float = float(inp_dim)
        w, h = orig_im.shape[1], orig_im.shape[0]
        scaling_factor = min(inp_dim_float/w, inp_dim_float/h)
        # -----------------------------------------------------------

        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        im_preds = [x for x in output if int(x[0]) == im_id]
        
        vc = Counter(objs)
        for obj_name in objs:
            if obj_name in ["car", "motorbike", "truck", "bicycle", "autorickshaw"]:
                vehicle_count += 1
                
        # --- NEW CODE: Drawing the actual bounding boxes on the image ---
        for pred in im_preds:
            cls_name = classes[int(pred[-1])]
            if cls_name in ["car", "motorbike", "truck", "bicycle", "autorickshaw"]:
                # Scale coordinates back to original image size
                x1, y1, x2, y2 = pred[1:5]
                x1 = (x1 - (inp_dim_float - scaling_factor*w)/2) / scaling_factor
                x2 = (x2 - (inp_dim_float - scaling_factor*w)/2) / scaling_factor
                y1 = (y1 - (inp_dim_float - scaling_factor*h)/2) / scaling_factor
                y2 = (y2 - (inp_dim_float - scaling_factor*h)/2) / scaling_factor
                
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                
                # Draw Box and Label
                cv2.rectangle(orig_im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(orig_im, cls_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
        # Add total count text to top corner and save
        cv2.putText(orig_im, f"Total Vehicles: {vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        out_path = os.path.join("outputs", f"processed_lane_{input_image_count}.jpg")
        cv2.imwrite(out_path, orig_im)
        # -----------------------------------------------------------------

        print('\033[1m' + f"Lane : {input_image_count} - Number of Vehicles detected : {vehicle_count}")

        if vehicle_count > 0:
            lane_count_list.append(vehicle_count)

        if vehicle_count > lane_with_higher_count:
            lane_with_higher_count = vehicle_count
            denser_lane = input_image_count

        print('\033[0m' +"           {:15} {}".format("Vehicle Type", "Count"))
        for key, value in sorted(vc.items()):
            if key in ["car", "motorbike", "truck", "bicycle", "autorickshaw"]:
                print('\033[0m' + f"            {key:15s} {value}")

    if CUDA:
        torch.cuda.synchronize()

if 'vehicle_count' not in locals() or vehicle_count == 0:
    print('\033[1m' + "There are no vehicles present from the input.")

print('\033[1m' + "-"*120)
print('🚥' + '\033[1m' + '\033[94m' + f" Lane with denser traffic is : Lane {denser_lane}" + '\033[30m' + "\n")

if lane_count_list:
    switching_time = avg_signal_oc_time(lane_count_list)
    switch_signal(denser_lane, switching_time)

print('\033[1m' + "-"*120)
try:
    output
except NameError:
    print("No detections were made | No Objects were found from the input")
    exit()

torch.cuda.empty_cache()