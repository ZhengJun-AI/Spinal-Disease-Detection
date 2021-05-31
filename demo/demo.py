import numpy as np
import json

import torch
from torchvision import transforms
from PIL import Image
import os
from demo_utils import getresult
import argparse
import logging

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Demo for showing results')
parser.add_argument('-n', '--npy', dest='npy', type=str, default='./label-all-npy',
                    help='Path of npy files')
parser.add_argument('-l', '--label', dest='label', type=str, default='./new_data-bbox-cls.json',
                    help='Path of label file with all classification ground truth')
parser.add_argument('-r', '--json', type=str, default='./result',
                    dest='res', help='Path of jpg results')
parser.add_argument('-y', '--yolo', type=str, default='./models/best.pt',
                    dest='yolo', help='Path of pretrained YOLO model')
parser.add_argument('-s', '--seven', type=str, default='./models/c1_best_model.pt',
                    dest='seven', help='Path of pretrained Resnet model for 7 clf')
parser.add_argument('-t', '--two', type=str, default='./models/c2_best_model.pt',
                    dest='two', help='Path of pretrained Resnet model for 2 clf')
parser.add_argument('-i', '--index', type=int, default=0,
                    dest='idx', help='Choose an index of all data(range from 0 to 199)')
parser.add_argument('--show_label', action='store_true', help='Choose to show labels on results or not')
par = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=par.yolo, force_reload=True)
    resnet_model1 = torch.load(par.seven).to(device) # for 7 categories
    resnet_model2 = torch.load(par.two).to(device) # for 2 categories
    logging.info('Loaded three models successfully!')
except:
    logging.info('Loading fail!')
    os._exit(0)

assert os.path.exists(par.label), f'{par.label} not exist!'
with open(par.label, "r") as f:
    labels = json.load(f)

labels = sorted(labels, key=lambda x: x['id'])
npy_path = par.npy
res_path = par.res

if not os.path.exists(res_path):
    os.mkdir(res_path)
    logging.info(f'Create {res_path} for storing results.')
    
pre_tag, lab_tag = getresult(device, yolo_model, resnet_model1, resnet_model2, 
              npy_path, labels, res_path, par.idx, show_label=par.show_label)
logging.info(f'\nlabel predictions: {pre_tag}\nlabel ground truth: {lab_tag}')