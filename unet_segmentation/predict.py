
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import cv2
import os
import numpy as np

from network import Network

test_img_dir = "dataset/test"
test_segmented_img_dir = "segmented_img/test"
if not os.path.exists(test_segmented_img_dir):
    os.makedirs(test_segmented_img_dir)
model_dir = "model"
height = 136
width = 296
threshold = 0.5
batch_size = 64
shuffle_batch = True
use_cuda = torch.cuda.is_available()

# load iris model
iris_model = Network()
iris_model.load_state_dict(torch.load(os.path.join(model_dir, "iris.pth")))
iris_model.eval() # test mode

# load sclera model
sclera_model = Network()
sclera_model.load_state_dict(torch.load(os.path.join(model_dir, "sclera.pth")))
sclera_model.eval() # test mode

def get_dataloader(X):
    dataloader = DataLoader(dataset=TestDataLoader(X),
                            batch_size=batch_size,
                            shuffle=shuffle_batch)
    return dataloader

def load_dataset():
    X = []
    for root, dirs, files in os.walk(test_img_dir):
        for name in files:
            X.append(os.path.join(root, name))
    return X

def read_img(img_dir):
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))
    return img

def forward_model(dataloader):
    global iris_model
    global sclera_model

    for id, (X, img_name) in enumerate(dataloader):  # batch
        X = Variable(X, requires_grad=False)
        if use_cuda:
            X = X.cuda()
            iris_model = iris_model.cuda()
            sclera_model = sclera_model.cuda()

        # forward
        segmented_iris_img = iris_model(X.float())  # [batch, 1, 136, 296]
        segmented_sclera_img = sclera_model(X.float())  # [batch, 1, 136, 296]

        # save segmented img
        save_segmented_img(segmented_iris_img, img_name, "iris")
        save_segmented_img(segmented_sclera_img, img_name, "sclera")

def save_segmented_img(segmented_img, img_name, iris_sclera):
    # segmented_img -> (batch, 1, 136, 296)
    segmented_img = segmented_img.data.numpy()
    img_name = list(img_name)
    for _id in range(segmented_img.shape[0]):  # batch
        img = segmented_img[_id, 0, :, :]  # (136, 296)
        img = binarize_img(img)

        # extract from original img
        org_img = read_img(os.path.join(test_img_dir, img_name[_id]))
        h, w, c = org_img.shape
        result_img = np.zeros((h, w, c), np.uint8)  # black img
        for _h in range(h):
            for _w in range(w):
                if img[_h, _w] == 1:
                    result_img[_h, _w, 0] = org_img[_h, _w, 0]  # R
                    result_img[_h, _w, 1] = org_img[_h, _w, 1]  # G
                    result_img[_h, _w, 2] = org_img[_h, _w, 2]  # B

        # save result_img
        cv2.imwrite(os.path.join(test_segmented_img_dir, iris_sclera + "_" + img_name[_id]), result_img)

def binarize_img(img):
    h, w = img.shape[0], img.shape[1]
    for _h in range(h):
        for _w in range(w):
            if img[_h, _w] < threshold:
                img[_h, _w] = 0
            else:
                img[_h, _w] = 1
    return img

class TestDataLoader(object):

    def __init__(self, X):
        self.X = X
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        img_name = os.path.basename(self.X[index])
        img = self.read_img(self.X[index]) # read img (134, 296, 3)
        img = self.transform_img(img) # transform image
        return img, img_name # [21, 1, 134, 296] -> (batch, channel, height, width)

    def read_img(self, img_dir):
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height))
        return img

if __name__ == '__main__':

    X = load_dataset()
    test_dataloader = get_dataloader(X)
    forward_model(test_dataloader)

    