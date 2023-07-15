# Synthesis-Augmentation-for-Thermal-Face-Recognition
利用跨模态数据增强的方法解决红外数据的稀缺问题，进而提升红外人脸识别模型的性能的方法

## Table of contents

* [CycleGAN](#CycleGAN)
* [Facenet](#Facenet)
* [Evaluate](#Evaluate)
* [Data](#Data)




## CycleGAN

基于CycleGAN的改进模型（加入MSELoss）进行跨模态的图像合成（即可见光生成红外）
详细介绍见CycleGAN目录


## Facenet

基于Facenet使用VggFace2预训练权重完成红外人脸识别模型再训练


## Evaluate

红外人脸识别模型评估，完成对SpeakingFaces中划分的测试集进行测试评估


## Data

相应数据地址在各目录下可见


