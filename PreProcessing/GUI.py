from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import json
import os

root = Tk()
root.title('GUI')
root.geometry('800x800')
root.resizable(0, 0)

picName = StringVar()
files = []
idx = 0
fileName = ""
jsonData = None


def checkPic(fn):
    fn = fn.split('.')
    if len(fn) < 2:
        return False
    if not fn[1] in ('npy'):
        return False
    return True


def checkIdx(x):
    if x < 0:
        return len(files) - 1
    if x >= len(files):
        return 0
    return x


def getImage(file):
    img = np.load(file).astype(np.uint8)
    return ImageTk.PhotoImage(Image.fromarray(img))


def callback():
    global picName, files, idx, fileName, picName, curPic
    fileName = filedialog.askdirectory()
    files = os.listdir(fileName)
    assert files
    file = os.path.join(fileName, files[idx])
    while not checkPic(file):
        idx = checkIdx(idx + 1)
        file = os.path.join(fileName, files[idx])
    picName.set(files[idx])
    canvas.delete('all')
    img = getImage(file)
    image = canvas.create_image(0, 0, anchor='nw', image=img)
    root.mainloop()


def getJson():
    global jsonData
    jsonFile = filedialog.askopenfilename(
        filetypes=[("json文件", [".json"])])
    with open(jsonFile, 'r') as f:
        jsonData = json.load(f)
        if not 'all' in jsonFile:
            jsonData = jsonData['data']


def getPoints(fn):
    fn = fn.split('.')[0]
    for item in jsonData:
        if item['id'] == fn:
            return item['point']
    return None


def nextOne(dx):
    global idx, files, fileName, picName
    idx = checkIdx(idx + dx)
    if not files:
        return
    file = os.path.join(fileName, files[idx])
    while not checkPic(file):
        idx = checkIdx(idx + dx)
        file = os.path.join(fileName, files[idx])
    picName.set(files[idx])
    canvas.delete('all')
    img = getImage(file)
    image = canvas.create_image(0, 0, anchor='nw', image=img)
    if jsonData:
        points = getPoints(files[idx])
        points = sorted(points, key=lambda x: x['coord'][1])
        for i in range(len(points)):
            point = points[i]
            xx, yy = point['coord'][0], point['coord'][1]
            canvas.create_oval(xx - 2, yy - 2, xx + 2, yy + 2, fill='red')
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

            canvas.create_line(xx - w / 2, yy - h / 2, xx - w / 2, \
                               yy + h / 2, xx + w / 2, yy + h / 2, xx + w / 2,
                               yy - h / 2, xx - w / 2, yy - h / 2, fill='red')

    root.mainloop()


Button(root, text="选择npy文件夹..", command=callback).pack()
Button(root, text="选择json文件..", command=getJson).pack()
Button(root, text="上一张", command=lambda: nextOne(-1)).pack()
Button(root, text="下一张", command=lambda: nextOne(1)).pack()
Label(root, textvar=picName).pack()

W, H = 780, 780
canvas = Canvas(root, width=W, height=H, bg='white')
canvas.pack()

root.mainloop()
