import albumentations as albu
import cv2
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset


class AnomalyDACON(Dataset):
    def __init__(self, img_size, img_paths, labels, mode='train'):
        self.mode = mode
        self.img_size = img_size
        self.img_paths = img_paths
        self.labels = labels
        self.base_transform_list = [
            albu.Resize(self.img_size, self.img_size),
            albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, interpolation=1, border_mode=0,
                                  value=0, p=0.5),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        ]
        if self.mode == 'train':
            self.transform = albu.Compose([
                albu.RandomResizedCrop(height=self.img_size, width=self.img_size,
                                       scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                                       interpolation=1, p=1.0),
                albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
                                      interpolation=1, border_mode=0, value=0, p=0.5),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.OneOf([
                    albu.CLAHE(clip_limit=2, p=1.0),
                    albu.Sharpen(p=1.0),
                    albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1),
                                                  p=1.0),
                ], p=0.25),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2()
            ])
        elif self.mode == 'valid':
            self.transform = albu.Compose([
                albu.Resize(self.img_size, self.img_size),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2()
            ])
        elif self.mode == 'test':
            self.transform = albu.Compose([
                albu.Resize(self.img_size, self.img_size),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2()
            ])
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(f'{img_path}')

        good_labels =[3, 9, 15, 21, 28, 33, 40, 45, 52, 55, 63, 68, 72, 76, 84]

        if self.mode in ['train', 'valid']:
            label = self.labels[idx]
            if self.mode in ['train'] and label in good_labels:  # good object
                self.transform = albu.Compose([
                    albu.Resize(self.img_size, self.img_size),
                    albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, interpolation=1,
                                          border_mode=0,
                                          value=0, p=0.5),
                    albu.HorizontalFlip(p=0.5),
                    albu.VerticalFlip(p=0.5),
                    albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                    albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                    ToTensorV2()
                ])

            img = self.transform(image=img)

            return {'path': img_path, 'img': img, 'label': label}

        else:
            img = self.transform(image=img)
            return {'path': img_path, 'img': img, }
