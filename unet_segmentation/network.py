
import torch
import torch.nn.functional as F

# 2 down - 1 middle - 2 up layer

class Network(torch.nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        #################### DOWN ####################
        # down_block1
        self.down_block1_conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.down_block1_bn1 = torch.nn.BatchNorm2d(16)
        self.down_block1_conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.down_block1_bn2 = torch.nn.BatchNorm2d(16)
        self.down_block1_conv3 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.down_block1_bn3 = torch.nn.BatchNorm2d(16)
        self.down_block1_max_pool = torch.nn.MaxPool2d(2, 2)
        self.down_block1_relu = torch.nn.ReLU()

        # down_block2
        self.down_block2_conv1 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.down_block2_bn1 = torch.nn.BatchNorm2d(32)
        self.down_block2_conv2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.down_block2_bn2 = torch.nn.BatchNorm2d(32)
        self.down_block2_conv3 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.down_block2_bn3 = torch.nn.BatchNorm2d(32)
        self.down_block2_max_pool = torch.nn.MaxPool2d(2, 2)
        self.down_block2_relu = torch.nn.ReLU()
        #################### DOWN ####################

        #################### MIDDLE ####################
        self.middle_block_conv1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.middle_block_bn1 = torch.nn.BatchNorm2d(64)
        self.middle_block_conv2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.middle_block_bn2 = torch.nn.BatchNorm2d(64)
        self.middle_block_conv3 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.middle_block_bn3 = torch.nn.BatchNorm2d(64)
        self.middle_block_relu = torch.nn.ReLU()
        #################### MIDDLE ####################

        #################### UP ####################
        # up_block1
        self.up_block1_up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_block1_conv1 = torch.nn.Conv2d(32 + 64, 32, 3, padding=1)
        self.up_block1_bn1 = torch.nn.BatchNorm2d(32)
        self.up_block1_conv2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.up_block1_bn2 = torch.nn.BatchNorm2d(32)
        self.up_block1_conv3 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.up_block1_bn3 = torch.nn.BatchNorm2d(32)
        self.up_block1_relu = torch.nn.ReLU()

        # up_block2
        self.up_block2_up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_block2_conv1 = torch.nn.Conv2d(16 + 32, 16, 3, padding=1)
        self.up_block2_bn1 = torch.nn.BatchNorm2d(16)
        self.up_block2_conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.up_block2_bn2 = torch.nn.BatchNorm2d(16)
        self.up_block2_conv3 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.up_block2_bn3 = torch.nn.BatchNorm2d(16)
        self.up_block2_relu = torch.nn.ReLU()
        #################### UP ####################

        #################### LAST CONV. ####################
        self.last_conv1 = torch.nn.Conv2d(16, 16, 3, padding=1)  # 16,16
        self.last_bn1 = torch.nn.BatchNorm2d(16)  # 16
        self.last_conv2 = torch.nn.Conv2d(16, 1, 1, padding=0)
        self.last_relu = torch.nn.ReLU()
        #################### LAST CONV. ####################

    def forward(self, x):
        # down_block1
        x1 = self.down_block1_relu(self.down_block1_bn1(self.down_block1_conv1(x)))
        #print("x1:", x1.shape)
        x1 = self.down_block1_relu(self.down_block1_bn2(self.down_block1_conv2(x1)))
        #print("x1:", x1.shape)
        x1 = self.down_block1_relu(self.down_block1_bn3(self.down_block1_conv3(x1)))
        #print("x1:", x1.shape)
        self.x1 = x1  # copy and crop
        x1 = self.down_block1_max_pool(x1)
        #print("x1:", x1.shape, self.x1.shape)

        # down_block2
        x2 = self.down_block2_relu(self.down_block2_bn1(self.down_block2_conv1(x1)))
        x2 = self.down_block2_relu(self.down_block2_bn2(self.down_block2_conv2(x2)))
        x2 = self.down_block2_relu(self.down_block2_bn3(self.down_block2_conv3(x2)))
        self.x2 = x2
        x2 = self.down_block2_max_pool(x2)
        #print("x2:", x2.shape, self.x2.shape)

        # middle_block
        x3 = self.middle_block_relu(self.middle_block_bn1(self.middle_block_conv1(x2)))
        x3 = self.middle_block_relu(self.middle_block_bn2(self.middle_block_conv2(x3)))
        x3 = self.middle_block_relu(self.middle_block_bn3(self.middle_block_conv3(x3)))
        #print("middle:", x3.shape)

        # up_block1
        x = self.up_block1_up_sampling(x3)
        #print("up1:", x.size())
        #print("x2:", self.x2.size())

        x = torch.cat((x, self.x2), dim=1)
        # print("cat:",x.shape)
        x = self.up_block1_relu(self.up_block1_bn1(self.up_block1_conv1(x)))
        x = self.up_block1_relu(self.up_block1_bn2(self.up_block1_conv2(x)))
        x = self.up_block1_relu(self.up_block1_bn3(self.up_block1_conv3(x)))
        # print("up1:", x.shape)

        # up_block2
        x = self.up_block2_up_sampling(x)
        x = torch.cat((x, self.x1), dim=1)
        x = self.up_block2_relu(self.up_block2_bn1(self.up_block2_conv1(x)))
        x = self.up_block2_relu(self.up_block2_bn2(self.up_block2_conv2(x)))
        x = self.up_block2_relu(self.up_block2_bn3(self.up_block2_conv3(x)))
        # print("up2:", x.shape)

        # last conv.
        x = self.last_relu(self.last_bn1(self.last_conv1(x)))
        x = self.last_conv2(x)
        # print("last:", x.shape)

        return F.sigmoid(x)

