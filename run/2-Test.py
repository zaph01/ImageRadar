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

    # start testing-loop
    for i,data in enumerate(zip(test_loader.dataset.indices,test_loader.dataset.dataset)):
    # data is composed of [radar_FFT, segmap,out_label,box_labels,image]
        inputs = torch.Tensor([data[1]]).to(device).float()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)
        # fd = "C:/Users/malwi/ImageRadar/Git/output"
        # save = open(fd,"w")
        # save.write(outputs)
        
        ## hmi = DisplayHMI(data[4], data[0],outputs,enc)

        ##cv2.imshow('ImRadNet')
        
        # Press Q on keyboard to  exit
        ##if cv2.waitKey(25) & 0xFF == ord('q'):
          ##  break
    ##save.close         
##cv2.destroyAllWindows()
if __name__=='__main__':   
    config = json.load(open("C:/Users/malwi/ImageRadar/Git/config/config.json"))
    main(config)