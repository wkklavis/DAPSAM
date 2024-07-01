from torch.utils import data
import numpy as np
import math
import os
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *

class PROSTATE_dataset(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=384, batch_size=None, img_normalize=True):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.target_size = (target_size, target_size)
        self.img_normalize = img_normalize
        self.image_pool, self.label_pool, self.name_pool = [], [], []
        self._read_img_into_memory()
        if batch_size is not None:
            iter_nums = len(self.image_pool) // batch_size
            scale = math.ceil(250 / iter_nums)
            self.image_pool = self.image_pool * scale
            self.label_pool = self.label_pool * scale
            self.name_pool = self.name_pool * scale

        print('Image Nums:', len(self.img_list))
        print('Slice Nums:', len(self.image_pool))

    def __len__(self):
        return len(self.image_pool)

    def __getitem__(self, item):
        img_path, slice = self.image_pool[item]
        img_sitk = sitk.ReadImage(img_path)
        img_npy = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        img_npy = self.preprocess(img_npy[slice])

        label_path, slice = self.label_pool[item]
        label_sitk = sitk.ReadImage(label_path)
        label_npy = sitk.GetArrayFromImage(label_sitk)
        label_npy = np.expand_dims(label_npy[slice], axis=0)

        if self.img_normalize:
            # img_npy = normalize_image_to_0_1(img_npy)
            for c in range(img_npy.shape[0]):
                img_npy[c] = (img_npy[c] - img_npy[c].mean()) / img_npy[c].std()
        label_npy[label_npy > 1] = 1
        return img_npy, label_npy, self.name_pool[item]

    def _read_img_into_memory(self):
        img_num = len(self.img_list)
        for index in range(img_num):
            img_file = os.path.join(self.root, self.img_list[index])
            label_file = os.path.join(self.root, self.label_list[index])

            img_sitk = sitk.ReadImage(img_file)
            label_sitk = sitk.ReadImage(label_file)

            img_npy = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
            label_npy = sitk.GetArrayFromImage(label_sitk)

            for slice in range(img_npy.shape[0]):
                if label_npy[slice, :, :].max() > 0:
                    self.image_pool.append((img_file, slice))
                    self.label_pool.append((label_file, slice))
                    self.name_pool.append(img_file)

    def preprocess(self, x):
        # x = img_npy[slice]
        mask = x > 0
        y = x[mask]

        lower = np.percentile(y, 0.2)
        upper = np.percentile(y, 99.8)

        x[mask & (x < lower)] = lower
        x[mask & (x > upper)] = upper
        return np.expand_dims(x, axis=0)
