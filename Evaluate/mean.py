# 定义要读取的文件列表
import csv
import numpy as np


name = 'infrared_dataSF_IR100+TF_IR112_15epoch_'
# name = 'infrared_datafacenet_100mix_'

# files = [name + f'experiment{i}.csv' for i in range(1, 11)]  # 更改i值为重复次数
files = [name + f'{i}.csv' for i in range(1, 11)]  # 更改i值为重复次数


# col_index = 4  # 第二列的索引为1
# row_index = 2  # 第一行的索引为0
def mae(col_index,row_index):
    # 存储要读取的数据的列表
    data_list = []

    # 遍历每个文件，读取数据并保存到列表中
    for file in files:
        with open(file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            # 读取第二列第一行的数据
            data = float(list(reader)[row_index][col_index])*100
            # 将数据添加到列表中
            data_list.append(data)


    # 计算均值和方差
    mean = np.mean(data_list)
    var = np.var(data_list)

    data_list.append(mean)
    data_list.append(var)
    print(data_list)

    # 输出均值和方差
    print("均值：", mean)
    print("方差：", var)

    # 将数据写入CSV文件
    with open('data-da23'+name+'.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(data_list)


# mae(4,2)  # N=1,cos
mae(4,11) # N=10,cos
# mae(5,2)  # N=1,ED
# mae(5,11)  # N=10,ED
# mae(6,2)   # N=1,err
mae(6,11)   # N=10,err

