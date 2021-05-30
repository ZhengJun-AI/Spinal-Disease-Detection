import json
import cv2
from skimage import util
import random
import os
import argparse
import logging
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Perform Data Augmentation')
parser.add_argument('-i', '--img', dest='img', type=str, default='./images',
                    help='Path of img files(.jpg)')
parser.add_argument('-j', '--json', type=str, default='./new_data-bbox-cls.json',
                    dest='json', help='Path of json file of all labels')
parser.add_argument('-r', '--result', dest='res', type=str, default='./result',
                    help='Path of augmentation results')
par = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

nums = np.zeros(7, ).astype(np.int)
numsv5 = np.zeros(2, ).astype(np.int)


def cut_image(img_path, js_path, res_path, sig):
    global nums, numsv5

    with open(js_path, 'r') as load_f:
        train_dict = json.load(load_f)
    train_dict = sorted(train_dict, key=lambda x: x['id'])

    error_name = []

    for i in train_dict:
        imname = i['id']
        impath = os.path.join(img_path, imname + '.jpg')
        if not os.path.exists(impath):
            continue
        img = cv2.imread(impath)

        i['point'] = sorted(i['point'], key=lambda x: x['coord'][1])
        point_num = 0
        for j in i['point']:
            try:
                x1, y1, x2, y2 = list(map(int, j['bbox']))
            except:
                logging.info(f'{imname} get wrong!')
                error_name.append(imname)

            cop_img = img[y1:y2, x1:x2]
            name = imname + '-' + str(point_num) + '.jpg'
            point_num += 1

            try:
                clf = int(j['class'][1])
                cv2.imwrite(os.path.join(sig[clf], str(nums[clf]) + '.jpg'), cop_img)
                nums[clf] += 1
            except:
                logging.info(f'ID {imname} with Wrong label: {j["class"]}!')
                continue

            try:
                clf = int(j['class'][3])
                cv2.imwrite(os.path.join(sig[clf + 7], str(numsv5[clf]) + '.jpg'), cop_img)
                numsv5[clf] += 1
            except:
                logging.info(f'ID {imname} with Wrong label: {j["class"]}!')
                continue


def center_move(img_path, js_path, res_path, sig):
    global nums, numsv5

    with open(js_path, 'r') as load_f:
        train_dict = json.load(load_f)
    train_dict = sorted(train_dict, key=lambda x: x['id'])

    for i in train_dict:
        imname = i['id']
        impath = os.path.join(img_path, imname + '.jpg')
        if not os.path.exists(impath):
            continue
        img = cv2.imread(impath)

        i['point'] = sorted(i['point'], key=lambda x: x['coord'][1])
        point_num = 0
        for j in i['point']:
            try:
                x1, y1, x2, y2 = list(map(int, j['bbox']))
                width = y2 - y1
                higth = x2 - x1
                if width / 10 > higth / 10:
                    move_step = int(width / 10)
                else:
                    move_step = int(higth / 10)
                if y1 - move_step < 0:
                    cop_img_up = img[0:y2 - move_step, x1:x2]
                else:
                    cop_img_up = img[y1 - move_step:y2 - move_step, x1:x2]
                cop_img_down = img[y1 + move_step:y2 + move_step, x1:x2]
                if x1 - move_step < 0:
                    cop_img_left = img[y1:y2, 0:x2 - move_step]
                else:
                    cop_img_left = img[y1:y2, x1 - move_step:x2 - move_step]
                cop_img_right = img[y1:y2, x1 + move_step:x2 + move_step]
                name = imname + '-' + str(point_num)
                point_num += 1

                try:
                    clf = int(j['class'][1])
                except:
                    logging.info(f'ID {imname} with Wrong label: {j["class"]}!')
                    continue

                cop_img = [cop_img_up, cop_img_down, cop_img_left, cop_img_right]
                for img in cop_img:
                    cv2.imwrite(os.path.join(sig[clf], str(nums[clf]) + '.jpg'), img)
                    nums[clf] += 1

                try:
                    clf = int(j['class'][3])
                except:
                    logging.info(f'ID {imname} with Wrong label: {j["class"]}!')
                    continue

                for img in cop_img:
                    cv2.imwrite(os.path.join(sig[clf + 7], str(numsv5[clf]) + '.jpg'), img)
                    numsv5[clf] += 1

            except:
                continue


def gaussNoise(img):
    noise_img = util.random_noise(img, mode='gaussian')
    result = util.img_as_ubyte(noise_img)

    return result


def sub_gauss(sig, clf, add_num, stable_num):
    global nums
    if stable_num == 0:
        return 0
    for i in range(add_num):
        choose_name = random.randint(0, stable_num)
        img_path = os.path.join(sig[clf], str(choose_name) + '.jpg')
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        result = gaussNoise(img)
        cv2.imwrite(os.path.join(sig[clf], str(nums[clf]) + '.jpg'), result)
        nums[clf] += 1


# ------------------main_function---------------------

img_path = par.img
json_path = par.json
res_path = par.res
sign = ['v_1', 'v_2', 'V1', 'V2', 'V3', 'V4', 'V5only', 'noV5', 'V5']
sig = [os.path.join(res_path, item) for item in sign]
if not os.path.exists(res_path):
    os.mkdir(res_path)
    for item in sig:
        os.mkdir(item)
cut_image(img_path, json_path, res_path, sig)
logging.info('Finish cutting!')
center_move(img_path, json_path, res_path, sig)
logging.info('Finish center point moving!')
num_dic = {item: nums[idx] for idx, item in enumerate(sign[:7])}
num_dic2 = {item: numsv5[idx] for idx, item in enumerate(sign[7:])}
logging.info(f'Show data about categories:\n {num_dic}, {num_dic2}')
max_class1 = max(num_dic, key=num_dic.get)
for idx, i in enumerate(num_dic.keys()):
    if i == max_class1:
        continue
    else:
        sub_gauss(sig, idx, abs(num_dic[max_class1] - num_dic[i]), num_dic[i])
max_class2 = max(num_dic2, key=num_dic2.get)
sub_gauss(sig[:-2], 1, abs(num_dic2[max_class2] - num_dic2['V5']), num_dic2['V5'])
logging.info('Done!')
