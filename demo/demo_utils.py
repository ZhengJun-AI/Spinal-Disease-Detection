import numpy as np
import json

import torch
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

def getresult(device, yolo_model, resnet_model1, resnet_model2, 
              npy_path, labels, res_path, idx, show_label=False):
    pre_l, lab_l, pre_tag, lab_tag, pred_center = [], [], [], [], []
    
    img_id = labels[idx]['id']
    image = np.load(os.path.join(npy_path, img_id + ".npy"))
    results = yolo_model(image, size=900)
    bbox = sorted(results.xyxy[0], key=lambda x:(x[5], -x[4]))
    bbox_gt = sorted(labels[idx]['point'], key=lambda x: x['coord'][1])
    
    new_label1 = [5,6,0,1,2,3,4]
    new_label2 = [1,0]
    label_dict1 = {0:'V1', 1:'V2', 2:'V3', 3:'V4', 4:'V5only', 5:'v_1', 6:'v_2'}
    label_dict2 = {0:'V5', 1:'noV5'}
    
    tf1 = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(0.3002, 0.1575)
    ])
    
    tf2 = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(0.3043, 0.1605)
    ])
    
    for i in range(len(bbox)):
        if i > 0 and bbox[i][5] == bbox[i-1][5]:
            continue
        
        box_index = list(map(int, bbox[i]))
        pred_center.append([(box_index[0]+box_index[2])/2, (box_index[1]+box_index[3])/2])
        image_box = image[box_index[1]:box_index[3],box_index[0]:box_index[2]]
        label = bbox_gt[box_index[5]]['class'].strip('()').split(',')
        
        try:
            label[0] = new_label1[int(label[0])]
            label[1] = new_label2[int(label[1])]
        except:
            continue
            
        label = torch.LongTensor(label).to(device)
        img1 = tf1(Image.fromarray(image_box)).repeat(3,1,1).unsqueeze(0).to(device)
        img2 = tf2(Image.fromarray(image_box)).repeat(3,1,1).unsqueeze(0).to(device)

        output1 = resnet_model1(img1)
        ret1, prediction1 = torch.max(output1.data, 1)

        output2 = resnet_model2(img2)
        ret2, prediction2 = torch.max(output2.data, 1)

        pre_l.append([prediction1.data.view_as(label[0]).cpu(),prediction2.data.view_as(label[0]).cpu()])
        lab_l.append([label[0].cpu(),label[1].cpu()])
    pre_tag = [[label_dict1[int(pre_l[i][0])],label_dict2[int(pre_l[i][1])]] for i in range(len(pre_l))]
    lab_tag = [[label_dict1[int(lab_l[i][0])],label_dict2[int(lab_l[i][1])]] for i in range(len(lab_l))]
    
    plt.figure(figsize=(12,8))
    plt.subplot(121)
    plt.title(f'{img_id + ".npy"} Ground Truth')
    plt.imshow(image)
    for i, item in enumerate(bbox_gt):
        x, y = item['coord']
        plt.plot(x, y, 'ro')
        if show_label:
            plt.text(x, y, lab_tag[i], color="w")
        
    plt.subplot(122)
    plt.title(f'{img_id + ".npy"} Prediction')
    plt.imshow(image)
    for i, item in enumerate(pred_center):
        x, y = item
        plt.plot(x, y, 'ro')
        if show_label:
            plt.text(x, y, pre_tag[i], color="w")
        
    plt.savefig(os.path.join(res_path, img_id + '.jpg'), dpi=200)
    plt.close('all')
    
    return pre_tag, lab_tag