
import torch
import cv2
from torchvision import transforms
import os

class MyDataloader(object):

    def __init__(self, X, y, height, width):
        self.X = X
        self.y = y
        self.height = height
        self.width = width
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_mask = transforms.Compose([
            transforms.Lambda(lambda x: x/255),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        img_name = os.path.basename(self.X[index])
        img = self.read_img(self.X[index]) # read img (134, 296, 3)
        img = self.transform_img(img) # transform image
        mask = self.read_mask(self.y[index]) # (134, 296)
        mask = self.transform_mask(mask)
        return img, mask, img_name # [21, 1, 134, 296] -> (batch, channel, height, width)

    def read_img(self, img_dir):
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.width, self.height))
        return img

    def read_mask(self, mask_dir):
        img = cv2.imread(mask_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.width, self.height))
        return img

        