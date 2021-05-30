import numpy as np
import os
import cv2
import pandas as pd
import json

ori = './label-all-npy'
files = [file.split('.')[0] for file in os.listdir(ori)]
res = './labels'
jsonPath = './label-all-data.json'
with open(jsonPath, 'r') as f:
    jsonData = json.load(f)

# csv_labels = open("csv_labels.csv", "w")


for idx, item in enumerate(jsonData):
    if item['id'] not in files:
        continue

    img_dir = os.path.join('./label-all-npy', item['id'] + '.npy')
    img = np.load(img_dir)
    imgh, imgw = img.shape

    points = item['point']
    # print(points)
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

        bbox = [max(xx - w / 2, 0), max(yy - h / 2, 0), min(xx + w / 2, imgw), min(yy + h / 2, imgh)]

        for j in range(11):
            if jsonData[idx]['point'][j]['identification'] == point['identification']:
                jsonData[idx]['point'][j]['bbox'] = bbox
        label = str(point['class'])
        filename = item['id'] + '.jpg'
        filename = os.path.join(res, filename)
        bbox = [str(i), str(xx), str(yy), str(w), str(h)]
        txt_label.write(' '.join(bbox))
        txt_label.write('\n')

    txt_label.close()
with open('data-bbox-cls.json', 'w') as f:
    json.dump(jsonData, f)
