
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os


# import random
# # 固定PyTorch的随机种子
# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
#
# # 固定Python的随机种子
# random.seed(0)
# np.random.seed(0)

for i in range(10):
    print("第%d次" % (int(i)+1))
    data_dir = '../data/SF_IR100'
    name = data_dir.split('/')[-1]
    batch_size = 32
    epochs = 15
    workers = 0 if os.name == 'nt' else 8
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    # dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    dataset = datasets.ImageFolder(data_dir, transform=trans)
    # 基于vggface2预训练权重训练人脸识别模型
    resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=len(dataset.class_to_idx)
    ).to(device)
    optimizer = optim.Adam(resnet.parameters(), lr=0.0001)
    scheduler = MultiStepLR(optimizer, [5, 10])


    # dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)
    img_inds = np.arange(len(dataset))
    np.random.shuffle(img_inds)
    train_inds = img_inds
    # train_inds = img_inds[:int(0.8 * len(img_inds))]
    # val_inds = img_inds[int(0.8 * len(img_inds)):]

    train_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_inds)
    )
    # val_loader = DataLoader(
    #     dataset,
    #     num_workers=workers,
    #     batch_size=batch_size,
    #     sampler=SubsetRandomSampler(val_inds)
    # )
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {
        'fps': training.BatchTimer(),
        'acc': training.accuracy
    }
    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    print('\n\nInitial')
    print('-' * 10)
    # resnet.eval()
    # training.pass_epoch(
    #     resnet, loss_fn, val_loader,
    #     batch_metrics=metrics, show_running=True, device=device,
    #     writer=writer
    # )

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        resnet.train()
        training.pass_epoch(
            resnet, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

        # resnet.eval()
        # training.pass_epoch(
        #     resnet, loss_fn, val_loader,
        #     batch_metrics=metrics, show_running=True, device=device,
        #     writer=writer
        # )

        # if (epoch+1) % 8 == 0:
        #     torch.save(resnet,"../data/experiment/multi_exp/facenet_"+name+"_"+str(epoch+1)+".pt")
    torch.save(resnet,"../data/model/SF_IR100/facenet_"+name+"_experiment"+str(i+1)+".pt")

    writer.close()






