

import itertools
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch
from statistics import mean
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

from mydataloader import MyDataloader
from network import Network

class Main(object):

    iris_sclera = "sclera"
    epoch = 100

    image_dir = "dataset/img"
    mask_dir = "dataset/mask"
    segmented_img_dir = "segmented_img"
    if not os.path.exists(segmented_img_dir):
        os.makedirs(segmented_img_dir)
    mask_img_type = ".tif"
    height = 136
    width = 296
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    batch_size = 64
    shuffle_batch = True
    lr = 0.00025
    threshold = 0.5
    use_cuda = torch.cuda.is_available()
    plot_dir = "plot"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    def __init__(self):
        pass

    @staticmethod
    def load_dataset():
        X, y = [], []
        for root, dirs, files in os.walk(Main.image_dir):
            for name in files:
                img_name = name.split(".")[0]
                X.append(os.path.join(root, name))
                y.append(os.path.join(Main.mask_dir, Main.iris_sclera + "_" + img_name + Main.mask_img_type))
        return X, y

    @staticmethod
    def get_dataloader(X, y):
        dataloader = DataLoader(dataset=MyDataloader(X, y, Main.height, Main.width),
                                batch_size=Main.batch_size,
                                shuffle=Main.shuffle_batch)
        return dataloader

    @staticmethod
    def IoU_area_score(pred, y):
        IoU = []
        y = y.data.numpy()
        pred = pred.data.numpy()
        for _id in range(y.shape[0]): # batch
            _gt = y[_id][:]
            _pred = np.array(list(map(lambda x: 0 if (x < Main.threshold) else 1, pred[_id][:]))) # map [0-1] probs to 0/1 discrete values
            intersect_pixel = sum(a == b == 1 for a, b in zip(_gt, _pred))
            gt_pixel = sum(a == 1 for a in _gt)
            IoU.append(float(intersect_pixel)/gt_pixel)
        return mean(IoU)

    @staticmethod
    def binarize_img(img):
        h, w = img.shape[0], img.shape[1]
        for _h in range(h):
            for _w in range(w):
                if img[_h, _w] < Main.threshold:
                    img[_h, _w] = 0
                else:
                    img[_h, _w] = 1
        return img

    @staticmethod
    def read_img(img_dir):
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (Main.width, Main.height))
        return img

    @staticmethod
    def save_segmented_img(segmented_img, img_name):
        #segmented_img -> (21, 1, 136, 296)
        segmented_img = segmented_img.data.numpy()
        img_name = list(img_name)
        for _id in range(segmented_img.shape[0]): # batch
            img = segmented_img[_id, 0, :, :] # (136, 296)
            img = Main.binarize_img(img)

            # extract from original img
            org_img = Main.read_img(os.path.join(Main.image_dir, img_name[_id]))
            h, w, c = org_img.shape
            result_img = np.zeros((h, w, c), np.uint8)  # black img
            for _h in range(h):
                for _w in range(w):
                    if img[_h, _w] == 1:
                        result_img[_h, _w, 0] = org_img[_h, _w, 0] # R
                        result_img[_h, _w, 1] = org_img[_h, _w, 1] # G
                        result_img[_h, _w, 2] = org_img[_h, _w, 2] # B

            # save result_img
            cv2.imwrite(os.path.join(Main.segmented_img_dir, Main.iris_sclera + "_" + img_name[_id]), result_img)

    @staticmethod
    def run_train(dataloader, model, criterion, optimizer):
        model.train()  # set train mode
        epoch_IoU = []
        epoch_loss = []
        for id, (X, y, img_name) in enumerate(dataloader):  # batch
            # convert tensors to variables
            X = Variable(X, requires_grad=False)
            y = Variable(y, requires_grad=False)
            if Main.use_cuda:
                X = X.cuda()
                y = y.cuda()
                model = model.cuda()

            outputs = model(X.float()) # [21, 136, 296]
            segmented_img = outputs # unet output
            outputs = outputs.view(outputs.size()[0], -1) # flat [21, 40256]
            y = y.view(y.size()[0], -1).float() # flat
            loss = criterion(outputs, y)
            epoch_loss.append(loss.item())  # save batch loss

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate IoU area
            batch_mean_IoU_score = Main.IoU_area_score(outputs, y)
            epoch_IoU.append(batch_mean_IoU_score)

            # save segmented img
            Main.save_segmented_img(segmented_img, img_name)

        return mean(epoch_loss), mean(epoch_IoU)

    @staticmethod
    def plot_loss_figure(train_loss, name):
        # to prevent overlap, clear the buffer
        plt.clf()

        plt.plot([i for i in range(1, Main.epoch + 1)], train_loss)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("loss graphic")
        plt.legend(["train"])
        # plt.show()
        plt.savefig(os.path.join(Main.plot_dir, Main.iris_sclera + "_" + name))

    @staticmethod
    def plot_IoU_figure(train_IoU, name):
        # to prevent overlap, clear the buffer
        plt.clf()

        plt.plot([i for i in range(1, Main.epoch + 1)], train_IoU)
        plt.xlabel("epoch")
        plt.ylabel("IoU")
        plt.title("IoU graphic")
        plt.legend(["train"])
        # plt.show()
        plt.savefig(os.path.join(Main.plot_dir, Main.iris_sclera + "_" + name))

    @staticmethod
    def run_pipeline():
        X, y = Main.load_dataset()
        train_dataloader = Main.get_dataloader(X, y)

        # model, loss, optimizer
        model = Network()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=Main.lr)

        train_loss = []
        train_IoU = []
        for epoch in range(1, Main.epoch + 1):
            print(epoch, " .epoch başladı ...")
            # train
            _train_loss, _train_IoU = Main.run_train(train_dataloader, model, criterion, optimizer)
            train_loss.append(_train_loss)
            train_IoU.append(_train_IoU)

            # info
            print("train loss -> ", _train_loss)
            print("train IoU -> ", _train_IoU)

            print(epoch, " .epoch bitti ...")

        # plot loss, IoU
        Main.plot_loss_figure(train_loss, "loss.png")
        Main.plot_IoU_figure(train_IoU, "IoU.png")

        # save model
        torch.save(model.state_dict(), os.path.join(Main.model_dir, Main.iris_sclera + ".pth"))


if __name__ == '__main__':

    Main.run_pipeline()

