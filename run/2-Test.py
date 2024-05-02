import sys
sys.path.append('C:/Users/malwi/ImageRadar/Clone/ImageRadar')
import os
import torch
import torch.nn as nn
import torch.nn.functional as f
import json
import torch.optim as optim
from loss.loss_function import pixor_loss
import numpy as np
import random
import argparse
#from torch.utils.tensorboard import SummaryWriter as SW
from pathlib import Path
from datetime import datetime
from torch.optim import lr_scheduler
from model.ImRadNet import ImRadNet
from model.ImRadNet import DataNorm
from dataset.dataloader import CreateDataLoaders
import cv2
##from utils.util import DisplayHMI
import dataset.dataloader as data_pcl
from dataset.dataloader import ImRad_PCL

def main(config, saved_model = 'ImRad.pth'):
    
    #set device -> CPU or GPU
    def get_device():
    # check if GPU is availabale, else use CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device
    device = get_device()

    #load dataset
    #########
    dataset = ImRad_PCL(root_dir = 'C:/Users/malwi/ImageRadar/Clone/ImageRadar/radar_PCL')
    train_loader,test_loader = data_pcl.CreateDataLoaders(dataset)
    #########   
    ######################
    ''''
    dataset = RADIal(root_dir = config['dataset']['root_dir'],
                        statistics= config['dataset']['statistics'],
                        encoder=enc.encode,
                        difficult=True)
    ######################
    '''
    #train_loader, test_loader = CreateDataLoaders(dataset,config['dataloader'],config['seed']
     
    #create model
    net = ImRadNet()

    Norm_Data = DataNorm()

    # move net to device
    net.to(device)

    # load the model
    ImRad = torch.load(saved_model)
    net.load_state_dict(ImRad['model_state_dict'])
    #net.load_state_dict(ImRad['optimizer_state_dict'])

    # set net to evaluation-mode
    net.eval()

    # Optimizer
    # lr = float(config['optimizer']['lr'])                           # define initial learning rate of first iteration
    # step_size = int(config['learning_rate']['step_size'])           # defines how many epochs should be run without changing the learning rate
    # gamma = float(config['learning_rate']['gamma'])                 # after the number of epochs defined in step_size, the learning rate is multiplied (in this case reduced) with the factor gamma
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    #freespace_loss = nn.BCEWithLogitsLoss(reduction='mean') 
    freespace_loss = nn.L1Loss()

    loss_summed = 0

    # start testing-loop
    for i,(data,labels) in enumerate(zip(test_loader.dataset.dataset,test_loader.dataset.indices)):
     #data is composed of [radar_FFT, segmap,out_label,box_labels,image]
        input = torch.Tensor([data[0:3]]).to(device).float()
        inputs = Norm_Data(input)
        running_loss = 0.0     
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
        # fd = "C:/Users/malwi/ImageRadar/Git/output"
        # save = open(fd,"w")
        # save.write(outputs)
        # if 'model' in config:
        #     if 'SegmentationHead' in config['model']:
        #         seg_map_label = torch.Tensor(labels[:]).to(device).double()
        # else:
        seg_map_label = torch.Tensor(labels[:]).to(device).double()

        prediction = outputs.contiguous().flatten()
        label = seg_map_label.contiguous().flatten()        
        loss = freespace_loss(prediction, label[0:3])

        if len(label) == 6: ## Two labels
            loss2 = freespace_loss(prediction,label[3:6])
            loss = min([loss,loss2]) 
        elif len(label) == 9: ## Three labels
            loss2 = freespace_loss(prediction,label[3:6])
            loss3 = freespace_loss(prediction,label[6:9])
            loss = min([loss,loss2,loss3])
        elif len(label) == 12: ## Four labels
            loss2 = freespace_loss(prediction,label[3:6])
            loss3 = freespace_loss(prediction,label[6:9])
            loss4 = freespace_loss(prediction,label[9:12])
            loss = min([loss,loss2,loss3,loss4])

        loss_summed += loss
    print(loss_summed/500)

        #scheduler.step()
        # losses 
        
##cv2.destroyAllWindows()
if __name__=='__main__':   
    config = json.load(open("C:/Users/malwi/ImageRadar/Clone/ImageRadar/config/config.json"))
    main(config)