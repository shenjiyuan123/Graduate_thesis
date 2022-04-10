import os
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from PIL import Image

class Deepfake_datasets(Dataset):
    def __init__(self, file_text_pth, transform=None, use_DCT=False):
        super().__init__()
        img_list = []
        with open(file_text_pth, 'r') as f:
            for line in f.readlines():
                img_list.append(line.strip().split('\t'))
        self.img_label_list = img_list
        self.transform = transform
        self.use_DCT = use_DCT

    def __len__(self):
        return len(self.img_label_list)

    def __getitem__(self, index):
        tmp, label = self.img_label_list[index]
        img = Image.open(tmp).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.use_DCT:
            img = self.DCT(img)

        label = int(label)
        return img, label

    def doDct(self, inputMatrix):
        fltMatrix = inputMatrix
        fltDct = cv2.dft(fltMatrix)
        return fltDct

    def DCT(self, img):
        raw_img = img.numpy()
        dct_img1 = self.doDct(raw_img[0,:,:])[np.newaxis,:,:]
        dct_img2 = self.doDct(raw_img[1,:,:])[np.newaxis,:,:]
        dct_img3 = self.doDct(raw_img[2,:,:])[np.newaxis,:,:]
        dct_img = np.concatenate((dct_img1,dct_img2,dct_img3),axis=0)
        return torch.from_numpy(dct_img)


transform = transforms.Compose([
    transforms.ToTensor()
])

def build_dataloader(dataset, batch_size, shuffle=True, num_workers=16, seed = 10):
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=np.random.seed(int(seed)),
        )
    return data_loader


if __name__=="__main__":
    train_dataset = Deepfake_datasets('/home/shenjiyuan/deepfake/deepfake/datasets/a.txt', \
                                      transform,True)
    train_dataloader = build_dataloader(train_dataset,2)
    for i, data in enumerate(train_dataloader):
        img, label = data[0], data[1]
        print(i, label, img.shape)
        break

