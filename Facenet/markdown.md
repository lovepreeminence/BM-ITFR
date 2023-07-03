基于Facenet使用VggFace2预训练权重完成红外人脸识别模型的训练

1、安装环境
选择已有虚拟环境或创建新的虚拟环境，在虚拟环境中安装运行Facenet所需要的库
所需库在./tests/travis_requirements.txt路径下

2、训练
（1）训练数据集
    +-- dataset_folder/
    |   +-- person1/
    |   |   +-- image1.jpg
    |   |   +-- image2.jpg
    |   |   +-- ...
    |   +-- person2/
    |   |   +-- image1.jpg
    |   |   +-- image2.jpg
    |   |   +-- ...
    |   +-- ...
其中，person1、person2等文件夹包含每个人的所有人脸图像，image1.jpg、image2.jpg等文件是对应的人脸图像文件。每个人的图像应该放在一个单独的文件夹中，并用该人的名称标识文件夹，即文件夹名为人名（label）。我们用数字1、2、...来命名每个人的，这样比较方便。

此外，还可能需要对数据进行预处理，将数据集里每个图像尺寸转为160X160。

（2）下载预训练权重VggFace2
因为我们是基于VggFace2预训练权重进行训练的，初次运行训练代码时会自动下载预训练权重，但由于网络原因可能会下载不成功导致无法运行，可以在https://github.com/timesler/facenet-pytorch中下载，然后存放在C:\Users\xxx\.cache\torch\checkpoints文件夹下。

（3）训练
代码facenet.py,训练集中没有划分验证集，同时由于小数据量下模型存在一定的波动，我们的针对同一个训练集，均训练了十次即十个模型，最终评估结果取十次均值并报告方差。
其中，data_dir = '../data/xxx'为训练集地址，模型保存地址为./data/model

训练完后测试评估在另外的代码中

