首先，我整合了原来提供的数据集，按照更加清晰明了的格式重新定义了json格式，然后储存为label-all-data.json。

同时，对于原有的dcm文件，直接提取像素矩阵储存为npy文件，既压缩了空间又方便了数据加载。事后看这一步可能显得有点多余，但是当时我在构思GUI工具的时候，第一时间想到的就是采用npy格式作为储存图像的格式。

随后使用test-npy.py生成了含有框的标注数据csv_label.csv，框的生成方式参考[第一名方案](https://img-blog.csdnimg.cn/img_convert/9cefbf463c5710a05fc628cd30a7ab4e.png)，分类编号参考自[第六名方案](https://img-blog.csdnimg.cn/img_convert/1c962f5d8253f996e2d7189edec0b8e3.png)。然后使用GUI.py可以看到框的具体样子，便于后期fine-tune。

最后就是使用csv2coco.py去生成COCO格式的数据集。至此为止，简单的数据处理就完成了，接下来就是要去跑模型。

在我看来，尽快把整个pipeline建立起来是很重要的，因此起始的时候采用简陋的架子是可以接受的。