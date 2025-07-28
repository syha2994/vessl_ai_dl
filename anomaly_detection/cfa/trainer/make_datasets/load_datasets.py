import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import cv2


class LoadDataset(Dataset):
    def __init__(self, dataset_path, resize=256, cropsize=256, info='train', train=True):
        self.dataset_path = dataset_path
        # self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.info = info

        self.x, self.y, self.mask = self.load_dataset_folder()

        if train:
            self.transform_x = T.Compose([
                T.ToTensor(),
                T.Resize((self.resize, self.resize), interpolation=T.InterpolationMode.BICUBIC),
                #T.Resize((self.resize, self.resize), interpolation=Image.Resampling.LANCZOS),
                T.RandomRotation(degrees=10), 
                T.RandomCrop((self.resize, self.resize), padding=4),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
                T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                T.RandomErasing(p=0.3, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=0),
                T.ElasticTransform(alpha=50.0, sigma=5.0),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        
            self.transform_mask = T.Compose([
                T.Resize((self.resize, self.resize), Image.NEAREST),
                T.ToTensor()
            ])
        else:
            self.transform_x = T.Compose([
                T.ToTensor(),
                T.Resize((self.resize, self.resize), interpolation=T.InterpolationMode.BICUBIC),
                #T.Resize((self.resize, self.resize), Image.Resampling.LANCZOS),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
            self.transform_mask = T.Compose([
                T.Resize((self.resize, self.resize), Image.NEAREST),
                T.ToTensor()
            ])





    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = self.info
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, phase)
        gt_dir = os.path.join(self.dataset_path, 'ground_truth')

        img_types = [f for f in sorted(os.listdir(img_dir)) if not f.startswith('.DS_Store')]
        # img_types = sorted(os.listdir(img_dir))
        print(img_types)
        for img_type in img_types:

            img_type_dir = os.path.join(img_dir, img_type)
            print(img_type_dir)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.bmp') or f.endswith('.png')])
            if phase == 'train':
                img_fpath_list = img_fpath_list
            x.extend(img_fpath_list)

            print(phase)

            if phase == 'real_test':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                if img_type == 'good':
                    y.extend([0] * len(img_fpath_list))
                    mask.extend([None] * len(img_fpath_list))
                else:
                    y.extend([1] * len(img_fpath_list))
                    gt_type_dir = os.path.join(gt_dir, img_type)
                    img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                     for img_fname in img_fname_list]
                    mask.extend(gt_fpath_list)
        y = list(map(int, y))

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
