import numpy as np
import os
import cv2
import pandas as pd
import json

ori = './label-all-npy'
files = os.listdir(ori)
res = './label-all-jpg'
jsonPath = './coco/annotations'


csv_labels = open("csv_labels.csv", "w")

for item in jsonData:
    points = item['point']
    print(points)
    points = sorted(points, key=lambda x: x['coord'][1])
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

        bbox = [str(xx - w / 2), str(yy - h / 2), str(xx + w / 2), str(yy + h / 2)]
        label = str(point['class'])
        filename = item['id'] + '.jpg'
        filename = os.path.join(res, filename)
        csv_labels.write(filename + "," + bbox[0] + "," + bbox[1] + "," + bbox[2] + "," + bbox[3] + "," + label + "\n")

csv_labels.close()
