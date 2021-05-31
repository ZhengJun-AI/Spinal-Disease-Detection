import numpy as np
import os
import json
import argparse
import logging

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Generate YOLO format labels')
parser.add_argument('-n', '--npy', dest='npy', type=str, default='./label-all-npy',
                    help='Path of npy files')
parser.add_argument('-l', '--label', dest='label', type=str, default='./labels',
                    help='Path of label file results')
parser.add_argument('-j', '--json', type=str, default='./label-all-data.json',
                    dest='json', help='Path of json file of all labels')
par = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

ori = par.npy
res = par.label
jsonPath = par.json

assert os.path.exists(ori), f'{ori} not exist!'
files = [file.split('.')[0] for file in os.listdir(ori)]
logging.info(f'Load {len(files)} npy files')

assert os.path.exists(jsonPath), f'{jsonPath} not exist!'
with open(jsonPath, 'r') as f:
    jsonData = json.load(f)

if not os.path.exists(res):
    os.mkdir(res)

for idx, item in enumerate(jsonData):
    if item['id'] not in files:
        continue

    img_dir = os.path.join('./label-all-npy', item['id'] + '.npy')
    assert os.path.exists(img_dir), f'{img_dir} not exist!'
    img = np.load(img_dir)

    imgh, imgw = img.shape

    points = item['point']
    points = sorted(points, key=lambda x: x['coord'][1])

    txt_dir = os.path.join(res, item['id'] + '.txt')
    txt_label = open(txt_dir, 'w')

    for i in range(len(points)):
        point = points[i]
        xx, yy = point['coord'][0], point['coord'][1]
        if i == 0:
            p1 = points[i + 1]
            x1, y1 = p1['coord'][0], p1['coord'][1]
            h = (y1 - yy) * 2
            w = h * 1.4
        if 0 < i < len(points) - 1:
            p1, p2 = points[i + 1], points[i - 1]
            x1, y1 = p1['coord'][0], p1['coord'][1]
            x2, y2 = p2['coord'][0], p2['coord'][1]
            h = max(y1 - yy, yy - y2) * 2
            w = h * 1.4
        if i == len(points) - 1:
            p2 = points[i - 1]
            x2, y2 = p2['coord'][0], p2['coord'][1]
            h = (yy - y2) * 2
            w = h * 1.4

        bbox = [i, max(xx - w / 2, 0) / imgw, max(yy - h / 2, 0) / imgh,
                min(xx + w / 2, imgw) / imgw, min(yy + h / 2, imgh) / imgh]
        bbox = list(map(str, bbox))
        txt_label.write(' '.join(bbox))
        txt_label.write('\n')

    txt_label.close()

logging.info('Process Successfully!')
