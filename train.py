import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import time
from tqdm import tqdm
from Models import * #model and dataset
import csv

#https://www.youtube.com/watch?v=BNPW1mYbgS4


def train(model,train_loader,loss_function,optimizer,device,trainlen,batch):
    model.train()
    train_loss=0
    train_accuracy=0.0
    TP=0.0
    FP=0.0
    FN=0.0
    for data, mask in tqdm(train_loader):
        data, mask = data.to(device), mask.to(device)
        optimizer.zero_grad()
        output=model(data.float())
        output=torch.sigmoid(output)
        loss=loss_function(output,mask.float())
        loss.backward()
        train_loss+=loss
        optimizer.step()
        #mask=torch.squeeze(mask,1)
        preds=(output>0.5).float()
        assert preds.size()==mask.size()
        train_accuracy+=int(torch.sum(preds==mask.data))
        TP+=int(torch.sum((preds==mask.data)*(preds==1)))
        FP+=int(torch.sum((preds!=mask.data)*(preds==1)))
        FN+=int(torch.sum((preds!=mask.data)*(preds==0)))
    train_accuracy=train_accuracy/(trainlen*256*256)
    train_loss=train_loss/trainlen*batch
    train_IOU=TP/(TP+FP+FN)
    train_dice=(2*TP)/((2*TP)+FP+FN)
    return train_accuracy,train_loss,train_IOU,train_dice

def valid(model,valid_loader,loss_function,device,validlen,batch):
    model.eval()
    valid_loss=0.0
    valid_accuracy=0.0
    TP,FP,FN=0.0,0.0,0.0
    with torch.no_grad():
        for j, (data, mask) in enumerate(valid_loader):
            data, mask = data.to(device), mask.to(device)
            output=model(data.float())
            output=torch.sigmoid(output)
            loss=loss_function(output,mask.float())
            valid_loss+=loss
            preds=(output>0.5).float()
            valid_accuracy+=int(torch.sum(preds==mask.data))
            TP+=int(torch.sum((preds==mask.data)*(preds==1)))
            FP+=int(torch.sum((preds!=mask.data)*(preds==1)))
            FN+=int(torch.sum((preds!=mask.data)*(preds==0)))
    valid_accuracy=valid_accuracy/(validlen*256*256)
    valid_loss=valid_loss/validlen*batch
    valid_IOU=TP/(TP+FP+FN)
    valid_dice=(2*TP)/((2*TP)+FP+FN)
    return valid_accuracy, valid_loss, valid_IOU, valid_dice

def save_prediction_and_report(test_loader,testlen,save_path,epoch,model,device='cuda:0',model_name = 'UNet',note = '',report_name='report.csv'):
    model.eval()
    test_accuracy=0.0
    TP,FP,FN=0.0,0.0,0.0
    with torch.no_grad():
        for j, (data, mask, filename) in enumerate(test_loader):
            data, mask = data.to(device), mask.to(device)
            output=model(data.float())
            preds=torch.sigmoid(output)
            preds=(preds>0.5).float()
            test_accuracy+=int(torch.sum(preds==mask.data))
            TP+=int(torch.sum((preds==mask.data)*(preds==1)))
            FP+=int(torch.sum((preds!=mask.data)*(preds==1)))
            FN+=int(torch.sum((preds!=mask.data)*(preds==0)))
            preds = 255 * preds
            #check how to save image, maybe make a dataset class for test
            torchvision.utils.save_image(preds,os.path.join(save_path,filename)) #need to change, try to get filename, or also save original image
    test_accuracy=test_accuracy/(testlen*256*256)
    test_IOU=TP/(TP+FP+FN)
    test_dice=(2*TP)/((2*TP)+FP+FN)

    to_csv=[model_name,epoch,str(test_accuracy),str(test_IOU),str(test_dice),note]

    if not os.file.exists(report_name):
        with open(report_name, 'a',newline='') as report:
            writer = csv.writer(report)
            header=['Model','Epoch','Accuracy','IOU','Dice','note']
            writer.writerow(header)
            writer.writerow(to_csv)
    else:
        with open(report_name, 'a',newline='') as report:
            writer = csv.writer(report)
            writer.writerow(to_csv)
    return

def save_mask(model_path,image_path,save_path,dev="cuda:0"):
    model = torch.load(model_path).to(dev)
    model.eval()
    imgs = os.listdir(image_path)
    for img in tqdm(imgs):
        datafile = Image.open(os.path.join(image_path,img)).convert("RGB")
        datafile = datafile.resize((256,256))
        to_tensor = transforms.ToTensor()
        datafile = to_tensor(datafile)
        datafile = datafile.unsqueeze(0).to(dev)
        output = torch.sigmoid(model(datafile))
        output = (output>0.5).float()
        output = 255 * output
        torchvision.utils.save_image(output, os.path.join(save_path,img))

def save_report(Model, Method, epoch, train_acc, train_IOU, train_Dice, train_loss, valid_acc, valid_IOU, valid_Dice, valid_loss,note='',filename='report.csv'):
    to_csv=[Model, Method, epoch, train_acc, train_IOU, train_Dice,train_loss, valid_acc, valid_IOU, valid_Dice,valid_loss,note]
    if not os.path.exists(filename):
        with open(filename, 'a',newline='') as report:
            writer = csv.writer(report)
            header=['Model','Method','Epoch','train_Acc','train_IOU','train_Dice','train_Loss','valid_Acc','valid_IOU','valid_Dice','valid_Loss','note']
            writer.writerow(header)
            writer.writerow(to_csv)
    else:
        with open(filename, 'a',newline='') as report:
            writer = csv.writer(report)
            writer.writerow(to_csv)
