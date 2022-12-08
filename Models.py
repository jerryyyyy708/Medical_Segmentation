import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision import transforms
import torchvision
import numpy as np
import os
import torchvision.transforms.functional as F

class DiceLoss(nn.Module):
    #tensor.sum https://stackoverflow.com/questions/44790670/torch-sum-a-tensor-along-an-axis
    #https://blog.csdn.net/CaiDaoqing/article/details/90457197
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self,input,target):
        N=target.size(0)
        smooth=1

        input_flat = input.view(N,-1)
        target_flat = target.view(N,-1)
        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + smooth + target_flat.sum(1))
        loss = 1 - loss.sum()/N

        return loss

class IOULoss(nn.Module):
    def __init__(self):
        super(IOULoss, self).__init__()
    
    def forward(self,input,target):
        N=target.size(0)
        smooth=1

        input_flat = input.view(N,-1)
        target_flat = target.view(N,-1)
        intersection = input_flat * target_flat

        loss = (intersection.sum(1)) / (input_flat.sum(1) + target_flat.sum(1) - intersection.sum(1))
        loss = 1 - loss.sum()/N

        return loss


class UNet_Drop(nn.Module):
    def __init__(self):
        super().__init__()

        n_channel = 3
        n_classes = 1
        self.in_conv = SuccessiveConv(n_channel, 64)
        self.down_1 = ContractingPath(64, 128)
        self.down_2 = ContractingPath(128, 256)
        self.down_3 = ContractingPath(256, 512)
        self.drop_1 = nn.Dropout(0.2)
        self.down_4 = ContractingPath(512, 1024)
        self.drop_2 = nn.Dropout(0.2)
        self.up_1 = ExpandingPath(1024, 512)
        self.up_2 = ExpandingPath(512, 256)
        self.up_3 = ExpandingPath(256, 128)
        self.up_4 = ExpandingPath(128, 64)
        #self.out_conv = SuccessiveConv(64, 1)
        self.conv_last = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x_in_conv = self.in_conv(x)
        x_down_1 = self.down_1(x_in_conv)
        x_down_2 = self.down_2(x_down_1)
        x_down_3 = self.down_3(x_down_2)
        x_down_3 = self.drop_1(x_down_3)
        x_down_4 = self.down_4(x_down_3)
        x_down_4 = self.drop_2(x_down_4)
        x_up_1 = self.up_1(x_down_4, x_down_3)
        x_up_2 = self.up_2(x_up_1, x_down_2)
        x_up_3 = self.up_3(x_up_2, x_down_1)
        x_up_4 = self.up_4(x_up_3, x_in_conv)
        #x_out_conv = self.out_conv(x_up_4)
        ret=self.conv_last(x_up_4)
        return ret#x_out_conv

class SuccessiveConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.successive_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.successive_conv(x)



class ExpandingPath(nn.Module):#plus sussfive to that
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels,kernel_size=2,stride=2)
        self.conv = SuccessiveConv(in_channels, out_channels)
    def forward(self,x1,x2):
        up_x = self.up_sample(x1)
        x = torch.cat((up_x,x2),dim=1)
        return self.conv(x)

class Images(Dataset):
    def __init__(self, img_path, mask_path):
        self.img_path = img_path
        self.mask_path = mask_path

        img_files = set(os.listdir(self.img_path))
        mask_files = set(os.listdir(self.mask_path))
        self.files = list(img_files & mask_files)

    def __getitem__(self, index):
        filename = self.files[index]
        img_file = Image.open(self.img_path + '/' + filename).convert("RGB")
        mask_file = Image.open(self.mask_path + '/' + filename).convert('L')
        img_file=img_file.resize((256,256))
        mask_file=mask_file.resize((256,256))
        mask_file= np.array(mask_file,dtype=np.float32)
        mask_file[mask_file!=0.0]=1.0
        to_tensor = transforms.ToTensor()
        return to_tensor(img_file), to_tensor(mask_file)
    
    def __len__(self):
        return len(self.files)



class test_images(Dataset):
    def __init__(self, img_path, mask_path):
        self.img_path = img_path
        self.mask_path = mask_path

        img_files = set(os.listdir(self.img_path))
        mask_files = set(os.listdir(self.mask_path))
        self.files = list(img_files & mask_files)

    def __getitem__(self, index):
        filename = self.files[index]
        img_file = Image.open(self.img_path + '/' + filename).convert("RGB")
        mask_file = Image.open(self.mask_path + '/' + filename).convert('L')
        img_file=img_file.resize((256,256))
        mask_file=mask_file.resize((256,256))
        mask_file= np.array(mask_file,dtype=np.float32)
        mask_file[mask_file!=0.0]=1.0
        to_tensor = transforms.ToTensor()
        return to_tensor(img_file), to_tensor(mask_file), filename
    
    def __len__(self):
        return len(self.files)


class encoder(nn.Module):
    def __init__(self,n_classes = 3):
        super().__init__()
        n_channel = 3
        self.in_conv = SuccessiveConv(n_channel, 64)
        self.down_1 = ContractingPath(64, 128)
        self.down_2 = ContractingPath(128, 256)
        self.down_3 = ContractingPath(256, 512)
        self.down_4 = ContractingPath(512, 1024)
        self.fc1 = nn.Linear(4*4*1024,n_classes)

    def forward(self, x):
        x_in_conv = self.in_conv(x)
        x_down_1 = self.down_1(x_in_conv)
        x_down_2 = self.down_2(x_down_1)
        x_down_3 = self.down_3(x_down_2)
        x_down_4 = self.down_4(x_down_3)
        x_down_4 = x_down_4.view(-1,1024*4*4)
        fc1 = self.fc1(x_down_4)
        return fc1

class pretrain_set(Dataset):
    def __init__(self, img_path):
        self.img_path = img_path
        self.files = os.listdir(self.img_path)

    def __getitem__(self, index):
        filename = self.files[index]
        img_file = Image.open(self.img_path + '/' + filename).convert("RGB")
        img_file=img_file.resize((256,256))
        label = index%8
        degree = [0,45,90,135,180,225,270,315]
        img_file = F.rotate(img_file,degree[label])
        to_tensor = transforms.ToTensor()
        return to_tensor(img_file), label
    
    def __len__(self):
        return len(self.files)

class MocoSet(Dataset):
    def __init__(self, img_path):
        self.img_path = img_path
        self.files = os.listdir(self.img_path)

    def __getitem__(self, index):
        filename = self.files[index]
        img_file = Image.open(self.img_path + '/' + filename).convert("RGB")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])
        return transform(img_file),transform(img_file)
    
    def __len__(self):
        return len(self.files)

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.VGG=torchvision.models.vgg19(pretrained = True)
        self.VGG.classifier[6] = nn.Linear(4096,2)
    def forward(self,x):
        return self.VGG(x)

class Double_UNet(nn.Module):
    #缺 squeeze and excitation block, 2/3 -> 1/2
    def __init__(self,pretrained = False):
        super().__init__()
        if pretrained:
            self.VGG = torch.load('vgg.pt').VGG
        else:
            self.VGG=torchvision.models.vgg19(pretrained = True)
        #VGG
        self.VGG_block1 = nn.Sequential(*self.VGG.features[:4])#64
        self.VGG_block2 = nn.Sequential(*self.VGG.features[4:9])#128
        self.VGG_block3 = nn.Sequential(*self.VGG.features[9:18])#256
        self.VGG_block4 = nn.Sequential(*self.VGG.features[18:27])#512
        self.VGG_block5 = nn.Sequential(*self.VGG.features[27:-1])
        #ASPP
        self.ASPP = ASPP()
        #Decoder1
        self.dec1_block1 = Decoder_Block(1024,512)
        self.dec1_block2 = Decoder_Block(512,256)
        self.dec1_block3 = Decoder_Block(256,128)
        self.dec1_block4 = Decoder_Block(128,64)
        self.dec1_conv = nn.Conv2d(64,1,1)
        #encoder 2
        self.enc2_block1 = Encoder_Block(3,64,first=True)
        self.enc2_block2 = Encoder_Block(64,128)
        self.enc2_block3 = Encoder_Block(128,256)
        self.enc2_block4 = Encoder_Block(256, 512)
        self.enc2_block5 = Encoder_Block(512, 512)
        #decoder2
        self.dec2_block1 = Decoder2_Block(1024,512)
        self.dec2_block2 = Decoder2_Block(512,256)
        self.dec2_block3 = Decoder2_Block(256,128)
        self.dec2_block4 = Decoder2_Block(128,64)
        self.dec2_conv = nn.Conv2d(64,1,1)

    def forward(self,x):
        out1 = self.VGG_block1(x)#3,64
        out2 = self.VGG_block2(out1)#64,128
        out3 = self.VGG_block3(out2)#128,256
        out4 = self.VGG_block4(out3)#256,512
        output = self.VGG_block5(out4)
        aspp_out = self.ASPP(output)
        d1_1 = self.dec1_block1(aspp_out,out4)
        d1_2 = self.dec1_block2(d1_1,out3)
        d1_3 = self.dec1_block3(d1_2,out2)
        d1_4 = self.dec1_block4(d1_3,out1)
        d1_output = self.dec1_conv(d1_4)
        x2 = torch.matmul(x,d1_output)
        out5 = self.enc2_block1(x2)
        out6 = self.enc2_block2(out5)
        out7 = self.enc2_block3(out6)
        out8 = self.enc2_block4(out7)
        output2 = self.enc2_block5(out8)
        aspp_out2 = self.ASPP(output2)
        d2_1 = self.dec2_block1(aspp_out2,out8,out4)
        d2_2 = self.dec2_block2(d2_1,out7,out3)
        d2_3 = self.dec2_block3(d2_2,out6,out2)
        d2_4 = self.dec2_block4(d2_3,out5,out1)
        d2_output = self.dec2_conv(d2_4)
        #final_output = torch.cat((d1_output,d2_output))
        return d2_output

class Decoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels,kernel_size=2,stride=2)
        self.conv = SuccessiveConv(in_channels, out_channels)
        self.se = SELayer(out_channels)
    def forward(self,x1,x2):
        up_x = self.up_sample(x1)
        x = torch.cat((up_x,x2),dim=1)
        return self.se(self.conv(x))

class Decoder2_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels,kernel_size=2,stride=2)
        self.conv = SuccessiveConv(in_channels//2*3, out_channels)
        self.se = SELayer(out_channels)
    def forward(self,x1,x2,x3):
        up_x = self.up_sample(x1)
        x = torch.cat((up_x,x2,x3),dim=1)
        return self.se(self.conv(x))

class Encoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels,first=False):
        super().__init__()
        if first:
            self.contracting_path = nn.Sequential(
                SuccessiveConv(in_channels, out_channels),
                SELayer(out_channels)
            )
        else:
            self.contracting_path = nn.Sequential(
                nn.MaxPool2d(2),
                SuccessiveConv(in_channels, out_channels),
                SELayer(out_channels)
            )
    
    def forward(self, x):
        return self.contracting_path(x)


class ASPP(nn.Module):
    #https://www.cnblogs.com/haiboxiaobai/p/13029920.html
    def __init__(self, in_channel=512, depth=1024):
        super(ASPP,self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear',align_corners=True)
 
    def forward(self, x):
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = self.upsample(image_features)
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

class UNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        n_channel = 3
        n_classes = 1
        if pretrained:
            encoder=torch.load('encoder.pt')
            self.in_conv = encoder.in_conv
            self.down_1 = encoder.down_1
            self.down_2 = encoder.down_2
            self.down_3 = encoder.down_3
            self.down_4 = encoder.down_4
            del encoder
        else:
            self.in_conv = SuccessiveConv(n_channel, 64)
            self.down_1 = ContractingPath(64, 128)
            self.down_2 = ContractingPath(128, 256)
            self.down_3 = ContractingPath(256, 512)
            self.down_4 = ContractingPath(512, 1024)
        self.up_1 = ExpandingPath(1024, 512)
        self.up_2 = ExpandingPath(512, 256)
        self.up_3 = ExpandingPath(256, 128)
        self.up_4 = ExpandingPath(128, 64)
        #self.out_conv = SuccessiveConv(64, 1)
        self.conv_last = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x_in_conv = self.in_conv(x)
        x_down_1 = self.down_1(x_in_conv)
        x_down_2 = self.down_2(x_down_1)
        x_down_3 = self.down_3(x_down_2)
        x_down_4 = self.down_4(x_down_3)
        x_up_1 = self.up_1(x_down_4, x_down_3)
        x_up_2 = self.up_2(x_up_1, x_down_2)
        x_up_3 = self.up_3(x_up_2, x_down_1)
        x_up_4 = self.up_4(x_up_3, x_in_conv)
        #x_out_conv = self.out_conv(x_up_4)
        ret=self.conv_last(x_up_4)
        return ret#x_out_conv

class ContractingPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.contracting_path = nn.Sequential(
            nn.MaxPool2d(2),
            SuccessiveConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.contracting_path(x)

class Polyp(Dataset):
    def __init__(self, img_path, mask_path):
        self.img_path = img_path
        self.mask_path = mask_path

        img_files = set(os.listdir(self.img_path))
        mask_files = set(os.listdir(self.mask_path))
        self.files = list(img_files & mask_files)

    def __getitem__(self, index):
        filename = self.files[index]
        img_file = Image.open(self.img_path + '/' + filename).convert("RGB")
        mask_file = Image.open(self.mask_path + '/' + filename).convert('L')
        img_file=img_file.resize((256,256))
        mask_file=mask_file.resize((256,256))
        mask_file= np.array(mask_file,dtype=np.float32)
        mask_file[mask_file!=0.0]=1.0
        to_tensor = transforms.ToTensor()
        sz = np.sum(mask_file)
        if sz>12282:
            label = 2
        elif sz>5973:
            label = 1
        else:
            label = 0
            
        return to_tensor(img_file),label 
    
    def __len__(self):
        return len(self.files)


class Polyp_or_Not(Dataset):
    def __init__(self, img_path):
        self.img_path = img_path

        img_files = set(os.listdir(self.img_path))
        self.files = list(img_files)

    def __getitem__(self, index):
        filename = self.files[index]
        img_file = Image.open(self.img_path + '/' + filename).convert("RGB")
        img_file=img_file.resize((64,64))
        to_tensor = transforms.ToTensor()
        if 'nopolyp_' in filename:
            label = 0
        else:
            label = 1
            
        return to_tensor(img_file),label 
    
    def __len__(self):
        return len(self.files)

class SELayer(nn.Module):
    #https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)