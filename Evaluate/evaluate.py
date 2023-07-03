# coding=utf-8
import os
from PIL import Image
from torchvision.transforms import ToTensor
from facenet_pytorch import InceptionResnetV1
import argparse
from tqdm import tqdm
import random
from tsne import *
import csv
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # solve problem:OMP:
import torch
import numpy as np
import torch.nn.functional as F


# name = 'facenet_da23_5400_100_'
name = 'SF_IR100+TF_IR112_15epoch_'
path = 'SF_IR100+TF_IR112(--)'
# model_list = [name + f'experiment{i}' for i in range(1, 11)]  # 更改i值为重复次数
model_list = [name + f'{i}' for i in range(1, 11)]  # 更改i值为重复次数
# 15



for j in model_list:
    print('模型名', j)


    def encode(imgs):  # aligned shape = (n_sample,3,160,160)
        # resnet_model = InceptionResnetV1(classify=False, pretraned='vggface2').eval().to(device)
        resnet_model = torch.load('../Facenet/data/model/Table3/' + path +'/'+ str(j) + '.pt',map_location='cuda:0').eval().to(device)
        with torch.no_grad():
            tensor = resnet_model(imgs)
            feature = tensor.detach().cpu()
        return feature


    def count_list(x):
        n = 0
        for i in x:
            for j in i:
                n += 1
        return n


    def split_list(x, n_split):
        n_sub = len(x) // n_split
        new_list = []
        n1 = 0
        n2 = n_split
        for i in range(n_sub):
            new_list.append(x[n1:n2])
            n1 += n_split
            n2 += n_split
        if n_split * n_sub < len(x):
            new_list.append(x[n_sub * n_split:])
        n_element = count_list(new_list)

        assert n_element == len(x)
        print('list split true')
        return new_list


    if __name__ == "__main__":
        # Training settings
        parser = argparse.ArgumentParser(description='Thermal face baseline')
        parser.add_argument('--data_path', type=str, default='./')
        parser.add_argument('--img_path', type=str, default='Datasets-gray')
        parser.add_argument('--enroll_cam_file_trails', default='/trails/enroll_cam_file_trails',
                            help='enroll camera img filepath to be facedetec and cropped')
        parser.add_argument('--enroll_infr_file_trails', default='/trails/enroll_infr_file_trails',
                            help='enroll infr img filepath to be facedetec and cropped')
        parser.add_argument('--test_cam_file_trails', default='/trails/test_cam_file_trails',
                            help='test camera img filepath to be facedetec and cropped')
        parser.add_argument('--test_infr_file_trails', default='/trails/test_infr_file_trails',
                            help='test infr img filepath to be facedetec and cropped')
        parser.add_argument('--train_cam_file_trails', default='/trails/train_cam_file_trails',
                            help='train camera img filepath to be facedetec and cropped')
        parser.add_argument('--train_infr_file_trails', default='/trails/train_infr_file_trails',
                            help='train infr img filepath to be facedetec and cropped')
        args = parser.parse_args()

        # kwargs = {'num_workers': 4, 'pin_memory': True}
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(device))
        img_pth = args.img_path
        trails_list = [
            # img_pth+args.enroll_cam_file_trails,
            img_pth + args.enroll_infr_file_trails,
            # img_pth+args.test_cam_file_trails,
            img_pth + args.test_infr_file_trails,
            # img_pth+args.train_cam_file_trails,
            # img_pth+args.train_infr_file_trails,
        ]

        # Process enroll_cam_file_trails
        for trail_name in trails_list:
            print('Process file:', trail_name)
            with open(args.data_path + trail_name, 'r') as f:
                file_trails = [args.img_path + (trail_name.strip('\n')).split(' ')[-1] for trail_name in f.readlines()]

            n_split = 40000  # list is too large, split to sublist incase memory out
            combine_sub_list = split_list(file_trails, n_split)
            print('total {} trails, split to {} sublist'.format(len(file_trails), len(combine_sub_list)))

            labels = []
            Features = []
            for i in range(len(combine_sub_list)):
                print('sublist={}……'.format(i))

                img_face_list = []
                for file_trail in tqdm(combine_sub_list[i]):
                    img = Image.open(file_trail).resize(
                        (160, 160))  # img cropped with facedetector; shape = (n_sample,3,160,160)
                    img_face_list.append(ToTensor()(np.array(img)))
                    label = file_trail.split('/')[-4] + '_' + file_trail.split('/')[-3] + '_' + \
                            (file_trail.split('/')[-1]).split('.')[0]
                    labels.append(label)
                # print(labels)
                print(len(labels))

                # img_combine to tensors
                img_faces = torch.stack(img_face_list)
                t_dataset = torch.utils.data.TensorDataset(img_faces)
                data_loader = torch.utils.data.DataLoader(t_dataset, batch_size=1000, shuffle=False)

                # img encoding
                print('Face feature extraction……')
                for batch_idx, data in enumerate(tqdm(data_loader)):
                    feature = encode(data[0].to(device))
                    Features.append(feature)

            Features = torch.cat(Features, 0)
            print('vector.shape=', Features.shape)
            print('n_labels=', len(labels))

            npz_file_name = (trail_name.split('/')[-1]).split('_')[0] + '_' + (trail_name.split('/')[-1]).split('_')[1]
            print('finally combined all sub list features!!! save {}.npz'.format(npz_file_name))
            np.savez('./Features/{}.npz'.format(npz_file_name), vector=Features, label=labels)


    def dataset_process(vectors, labels, M):
        spk_utt = {}
        for i in range(len(labels)):
            label_i = labels[i].split('_')[0] + '_' + labels[i].split('_')[1] + '_' + labels[i].split('_')[2] + '_' + \
                      labels[i].split('_')[3]

            # Speakingfaces-fake-1
            # label_i = labels[i].split('_')[0] + '_' + labels[i].split('_')[1] + '_' + labels[i].split('_')[2]

            # print(labels[1])
            if label_i not in spk_utt:
                spk_utt[label_i] = []
                spk_utt[label_i].append(vectors[i])
            else:
                spk_utt[label_i].append(vectors[i])

        if M == 'avg':
            # n_frame average
            vector = []
            labels_digt = []
            for key in spk_utt:
                vector.append(np.array(spk_utt[key]).mean(0))
                labels_digt.append(int(key.split('_')[1]))

        elif M == '1_frame':
            # 1_frame random select one frame as enroll
            vector = []
            labels_digt = []
            for key in spk_utt:
                i = random.randint(0, len(spk_utt[key]) - 1)
                vector.append(np.array(spk_utt[key])[i])
                labels_digt.append(int(key.split('_')[1]))

        return np.array(vector), np.array(labels_digt)


    if __name__ == "__main__":
        # Training settings
        parser = argparse.ArgumentParser(description='Thermal face baseline')
        # parser.add_argument('--data_path',   type=str,	default='/work9/cslt/caiyq/IR_SPK/THS2021/Dataset')
        parser.add_argument('--enroll_cam', default='./Features/enroll_cam.npz', help='enroll camera frame feature')
        parser.add_argument('--enroll_infr', default='./Features/enroll_infr.npz', help='enroll infrared frame feature')
        parser.add_argument('--test_infr', default='./Features/test_infr.npz', help='test infrared frame feature')
        parser.add_argument('--test_cam', default='./Features/test_cam.npz', help='test camera frame feature')
        parser.add_argument('--train_cam', default='./Features/train_cam.npz', help='train camera frame feature')
        parser.add_argument('--train_infr', default='./Features/train_infr.npz', help='train infrared frame feature')
        args = parser.parse_args()

        features_list = [
            # args.enroll_cam,
            args.enroll_infr,
            # args.test_cam,
            args.test_infr,
            # args.train_cam,
            # args.train_infr,
        ]

        # 1-frame feature process
        for fea_path in features_list:
            data = np.load(fea_path)  # load npz data
            vectors = data['vector']
            labels = data['label']

            vector, labels_digt = dataset_process(vectors, labels, '1_frame')  # '1_frame' or 'avg'
            print('\nvector.shape', vector.shape)
            print('n_labels=', labels_digt.shape)
            print('n_spks= ', len(set(labels_digt)))

            npz_file_name = (fea_path.split('/')[-1]).split('.')[0]
            np.savez('./Features_1f/{}.npz'.format(npz_file_name), vector=vector, label=labels_digt)

            # tsne plot dataset
            dirname = 'tsne_1f'
            file_name = (fea_path.split('/')[-1]).split('.')[0] + '.jpg'
        # if not os.path.exists("./{}/t-SNE-{}".format(dirname, file_name)):
        # tsne_plot_embedding(vector, labels_digt, file_name, dirname)

    def data_normlize(data):
        return (data - data.mean(0)) / data.std(0)


    def compute_eer(target_scores, nontarget_scores):
        if isinstance(target_scores, list) is False:
            target_scores = list(target_scores)
        if isinstance(nontarget_scores, list) is False:
            nontarget_scores = list(nontarget_scores)
        target_scores = sorted(target_scores)
        nontarget_scores = sorted(nontarget_scores)
        target_size = len(target_scores);
        nontarget_size = len(nontarget_scores)
        for i in range(target_size - 1):
            target_position = i
            nontarget_n = nontarget_size * float(target_position) / target_size
            nontarget_position = int(nontarget_size - 1 - nontarget_n)
            if nontarget_position < 0:
                nontarget_position = 0
            if nontarget_scores[nontarget_position] < target_scores[target_position]:
                break
        th = target_scores[target_position];
        eer = target_position * 1.0 / target_size;
        return eer, th


    def supervise_mean_var(data, label):
        assert (data.shape[0] == label.shape[0]), 'data and label must have the same length'
        label_class = np.array(list(set(label.numpy())))
        label_class = torch.from_numpy(label_class)
        mean_list = []
        lb_list = []
        for lb in label_class:
            data_j = data[label == lb]
            mean_j = torch.mean(data_j, 0, True)
            mean_list.append(mean_j)
            lb_list.append(torch.tensor([lb]))
        class_mean = torch.cat(mean_list, 0)
        class_label = torch.cat(lb_list)
        return class_mean, class_label


    def Cosine(probe_data, probe_label, test_data, test_label):
        probe_data, probe_label = torch.from_numpy(probe_data), torch.from_numpy(probe_label)
        test_data, test_label = torch.from_numpy(test_data), torch.from_numpy(test_label)
        class_mean, class_label = supervise_mean_var(probe_data, probe_label)

        Cosine_similaritys = []
        target_score = []
        no_target_score = []
        for i in range(len(test_data)):
            Cosine_similarity = F.cosine_similarity(test_data[i].unsqueeze(0), class_mean)
            Cosine_similaritys.append(Cosine_similarity.unsqueeze(0))

            for j in range(len(class_label)):
                if test_label[i] == class_label[j]:
                    target_score.append(Cosine_similarity.numpy().tolist()[j])
                else:
                    no_target_score.append(Cosine_similarity.numpy().tolist()[j])

        Cos_matix = torch.cat(Cosine_similaritys, 0)
        probe_index = Cos_matix.argmax(dim=1)
        predict_label = torch.index_select(class_label, 0, probe_index)

        label_mask = torch.eq(predict_label, test_label)
        accuracy = torch.true_divide(label_mask.sum(), len(label_mask))

        eer, th = compute_eer(target_score, no_target_score)
        return float('%.4f' % accuracy.numpy()), eer


    def Dist(probe_data, probe_label, test_data, test_label):
        probe_data, probe_label = torch.from_numpy(probe_data), torch.from_numpy(probe_label)
        test_data, test_label = torch.from_numpy(test_data), torch.from_numpy(test_label)

        dist_matix = torch.cdist(test_data, probe_data, p=2)
        probe_index = dist_matix.argmin(dim=1)
        predict_label = torch.index_select(probe_label, 0, probe_index)

        label_mask = torch.eq(predict_label, test_label)
        accuracy = torch.true_divide(label_mask.sum(), len(label_mask))
        return float('%.3f' % accuracy.numpy())


    def enroll_split(Features, labels, n_enroll):
        zz = list(zip(Features, labels))
        # shuffle(zz)
        enroll = []
        enroll_label = {}
        for i in zz:
            if i[1] not in enroll_label:
                enroll_label[i[1]] = 1
                enroll.append(i)
            elif enroll_label[i[1]] < n_enroll:
                enroll_label[i[1]] += 1
                enroll.append(i)
        enroll_data = np.array(list(zip(*enroll))[0])
        enroll_label = np.array(list(zip(*enroll))[1])
        return enroll_data, enroll_label


    if __name__ == "__main__":

        # camera & infrared data scoring; two kinds of feature processing
        header = ['n_enroll', 'camera', 'cos_acc', 'ED_acc', 'cos_eer', 'Infrared']
        csv1 = "infrared_data" + j + ".csv"
        # 将数据写入CSV文件
        with open(csv1, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['', header[1], '', '', '', header[5], '', '', ''])
            writer.writerow([header[0], header[2], header[3], header[4], header[2], header[3], header[4]])
        # for i in ['Features_1f','Features_avgf']:
        for i in ['Features_1f']:
            fea_path = './' + i
            print(i)

            # infrared data
            data = np.load(fea_path + '/enroll_infr.npz')  # load npz enroll data
            enroll_infr_data = data['vector']
            enroll_infr_label = data['label']
            data = np.load(fea_path + '/test_infr.npz')  # load npz test data
            test_infr_data = data['vector']
            test_infr_label = data['label']
            for n_enroll in range(1, 11):  # the number of enroll samples
                enroll_infr_data_n, enroll_infr_label_n = enroll_split(enroll_infr_data, enroll_infr_label, n_enroll)
                Cosine_accuracy, eer = Cosine(enroll_infr_data_n, enroll_infr_label_n, test_infr_data, test_infr_label)
                Distance_accuracy = Dist(enroll_infr_data_n, enroll_infr_label_n, test_infr_data, test_infr_label)
                print('Infrared: n_enroll={} cos_acc={} ED_acc={} cos_eer={}'.format(n_enroll, Cosine_accuracy,                                                                                   Distance_accuracy, eer))

                # 定义表头和数据
                # header = ['n_enroll', 'Infrared', 'cos_acc', 'ED_acc', 'cos_eer']
                data = [n_enroll, 'Infrared', Cosine_accuracy, Distance_accuracy, eer]

                # 将数据写入CSV文件
                with open(csv1, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['', '', '', '', data[2], data[3], data[4]])

            print('CSV文件已生成')


