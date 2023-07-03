基于CycleGAN的改进模型（加入了MSE Loss）进行跨模态的图像合成（即实现由可见光转换为红外）

1、安装坏境
选择已有虚拟环境或创建新的虚拟环境，在虚拟环境中安装运行CycleGAN所需要的库
在代码文件中requirements.txt文件里有所需的库，可以选择需要的逐个安装（pip install -r 库名），也可一次性安装（pip install -r requirements.txt）
若下载慢，可以在后面添加镜像源网址（https://blog.csdn.net/lian740930980/article/details/11043113），如pip install -r 库名 -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple

2、训练代码train.py
（1）训练部分需要pair（对齐配对的）的可见光-红外数据集，数据集的格式为：
    +-- dataset_folder/
    |   +-- trainA/
    |   |   +-- image1.jpg
    |   |   +-- image2.jpg
    |   |   +-- ...
    |   +-- trainB/
    |   |   +-- image1.jpg
    |   |   +-- image2.jpg
    |   |   +-- ...
    |   +-- testA/
    |   |   +-- image1.jpg
    |   |   +-- image2.jpg
    |   |   +-- ...
    |   +-- testB/
    |   |   +-- image1.jpg
    |   |   +-- image2.jpg
    |   |   +-- ...
其中，trainA和trainB文件夹分别包含两个不同域的训练图像，例如，想将可见光的图像转换成红外的图像，trainA文件夹中应该包含可见光的图像，trainB文件夹中应该包含红外的图像。同样的，testA和testB文件夹分别包含两个不同域的测试图像。

（2）训练参数设置
运行train.py代码时，参数设置有三种方式：
一是在Pycharm本地终端里，输入python train.py --dataroot ./datasets/xxx --name xxx --model cycle_gan即可运行，同时需要修改什么参数就在后面添加，--xxx样式为参数名，空一格在其后为参数内容（这一种比较麻烦，每一次运行都需要填写参数）
二是在train.py代码页面，右键点击-->更多运行/调试-->修改运行配置-->形参，在形参里输入--dataroot ./datasets/xxx --name xxx --model cycle_gan，应用并确定即可保存，点击运行即可
三是在./options/train_options.py里面修改参数配置（不推荐）

我们的参数设置为：
--dataroot ./datasets/dataset_folder --model cycle_gan --gpu_ids 0 --batch_size 10 --display_id 0 --n_epochs 50 --n_epochs_decay 50 --lr 0.0001 --beta1 0.1  --load_size 128 --crop_size 128 --name 5400_128_mse_bs10 --no_flip --serial_batches
其中，--dataroot即训练集路径，--n_epochs、--n_epochs_decay即设置的训练轮次（两者相加为总轮次），--name即训练出的模型名称，这三个可修改

训练完成后，我们在./checkpoints/路径下就得到了训练出的转换模型（以文件夹形式存放），里面有latest_net_G_A.pth（A域转B域的生成模型）、latest_net_G_B.pth（B域转A域的生成模型）、test_opt.txt（此次训练的参数设置）、web（训练过程图像）

3、测试代码test.py
（1）测试部分需要可见光数据（测试数据集格式在上面，训练、测试数据在一个文件夹中），通过生成模型G_A将可见光数据转为红外数据

（2）测试参数设置
测试参数修改同上
我们的参数设置：
--dataroot ./datasets/dataset_folder/testA --gpu_ids 0 --batch_size 10 --load_size 128 --crop_size 128 --name 5400_128_mse_bs10 --results_dir ./test_result/ --netG resnet_9blocks --model test  --no_dropout --model_suffix _A
其中，--dataroot即测试集路径，--name即选择训练好的模型，--results_dir即测试结果路径，--model test即单侧测试

4、训练好的G_A模型保存服务器地址/work9/kmust/SA4TFR/Code/CycleGAN/checkpoints/


