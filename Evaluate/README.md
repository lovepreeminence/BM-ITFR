# 红外人脸识别模型评估，用于对SpeakingFaces中划分的测试集进行测试评估

## Table of contents

* [数据预处理](#table-of-contents)
* [提取特征、打分](#quick-start)
* [计算均值方差](#pretrained-models)


## 数据预处理

按图片名划分各自的文件夹-->对每个文件夹里的图片重命名为：“infrared_f1.jpg”、“infrared_f2.jpg”、...-->在测试集中划分注册集，剩下的才为测试集（通过图片文件的相对路径名来划分的，注册集、测试集的路径地址存放在文件夹下）

数据集SF_IR42存放地址链接：https://pan.baidu.com/s/1eoflEVRVorR21jbniz-oyQ?pwd=qq2f 

## 提取特征、打分

数据集SF_IR42存放地址：./Datasets-gray/，其注册集、测试集的路径地址文件存放存放地址：./trails/
我们已写好一次对十个模型连续进行评估打分的代码，每次运行更换模型的地址即可


## 计算均值方差

将上面评估的十次结果进行计算，最终结果会保存至csv表格中
