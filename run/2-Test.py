import sys
sys.path.append('C:/Users/malwi/ImageRadar/Git')
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
    dataset = ImRad_PCL(root_dir = 'C:/Users/malwi/ImageRadar/Git/radar_PCL')
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

    # move net to device
    net.to(device)

    # load the model
    ImRad = torch.load(saved_model)
    net.load_state_dict(ImRad['model_state_dict'])

    # set net to evaluation-mode
    net.eval()

    # Optimizer
    lr = float(config['optimizer']['lr'])                           # define initial learning rate of first iteration
    step_size = int(config['learning_rate']['step_size'])           # defines how many epochs should be run without changing the learning rate
    gamma = float(config['learning_rate']['gamma'])                 # after the number of epochs defined in step_size, the learning rate is multiplied (in this case reduced) with the factor gamma
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    freespace_loss = nn.BCEWithLogitsLoss(reduction='mean') 

    # start testing-loop
    for i,data in enumerate(zip(test_loader.dataset.indices,test_loader.dataset.dataset)):
     #data is composed of [radar_FFT, segmap,out_label,box_labels,image]
        inputs = torch.Tensor([data[1]]).to(device).float()
        running_loss = 0.0     
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
        # fd = "C:/Users/malwi/ImageRadar/Git/output"
        # save = open(fd,"w")
        # save.write(outputs)
        if 'model' in config:
            if 'SegmentationHead' in config['model']:
                seg_map_label = torch.Tensor(data[2]).to(device).double()
        else:
            seg_map_label = torch.Tensor(data[1]).to(device).double()

        prediction = outputs.contiguous().flatten()
        label = seg_map_label.contiguous().flatten()        
        loss_seg = freespace_loss(prediction, label)
        loss_seg *= inputs.size(0)
        
        print(loss_seg)

        #scheduler.step()
        # losses 
        
##cv2.destroyAllWindows()
if __name__=='__main__':   
    config = json.load(open("C:/Users/malwi/ImageRadar/Git/config/config.json"))
    main(config)