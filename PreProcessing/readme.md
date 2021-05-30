# 数据处理部分

## 检测部分(yolo)

首先，我整合了原来提供的数据集，按照更加清晰明了的格式重新定义了json格式，然后储存为label-all-data.json。

同时，对于原有的dcm文件，直接提取像素矩阵储存为npy文件，既压缩了空间又方便了数据加载。事后看这一步可能显得有点多余，但是当时我在构思GUI工具的时候，第一时间想到的就是采用npy格式作为储存图像的格式。GUI程序运行效果如下所示：

<img src=".\GUIexample.png" alt="GUIexample" style="zoom:50%;" />

选取YOLOv5作为检测模型，利用npy2yolo.py生成yolo格式的标签数据。框的生成方式参考[第一名方案](https://img-blog.csdnimg.cn/img_convert/9cefbf463c5710a05fc628cd30a7ab4e.png)，分类编号参考自[第六名方案](https://img-blog.csdnimg.cn/img_convert/1c962f5d8253f996e2d7189edec0b8e3.png)。然后使用GUI.py可以看到框的具体样子，便于分析数据。

将文件整理成yolo格式(如下所示)，至此为止，简单的数据处理就完成了，接下来就是要去跑模型。

```tex
└── yolo-data
    ├── images
    |	└── train
    |	└── val
    |	└── test
    └── labels
    	└── train
    	└── val
    	└── test
```

