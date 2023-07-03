import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.A_paths = sorted(make_dataset(self.dir_A))
        print(self.A_paths,'A-------------------')
        self.B_paths = sorted(make_dataset(self.dir_B))
        print(self.B_paths, 'B-------------------')

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        print(self.A_size, self.B_size,'B-------------------')
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        # print(A_path,  'A-------------------')
        # print(B_path, 'B-------------------')
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A = self.transform(A_img)
        B = self.transform(B_img)
        return {'A': A, 'B': B,'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)