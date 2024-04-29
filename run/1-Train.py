############################### IMPORTANT INFORMATION ###################################
# This code contains functions from RADIal-Repository by Valeo.ai                       #  
# For further information go visit https://github.com/valeoai/RADIal?tab=readme-ov-file #
#########################################################################################
import os
import sys
## sys.path.append('C:/Users/mail/OneDrive/Dokumente/ImRad')
sys.path.append("C:/Users/mail\OneDrive - bwedu/Semester 5 (6)/Entwicklungsprojekt/Clone_29_04/ImageRadar")

import torch
import torch.nn as nn
import torch.nn.functional as f
import json
import torch.optim as optim
import numpy as np
import random
import argparse
from torch.utils.tensorboard import SummaryWriter as SW
from pathlib import Path
from datetime import datetime
from torch.optim import lr_scheduler
from model.ImRadNet import ImRadNet
from model.ImRadNet import DataNorm
from dataset.dataloader import CreateDataLoaders
from dataset.dataloader import ImRad_PCL
from loss.loss_function import pixor_loss
import dataset.dataloader as data_pcl



def main(config):      
    #input args: 
        #config = config.json
        #resume = "Speicherstand" des models -> resume training after going into validation-phase
    
    #set random seed
    torch.manual_seed(config['general']['seed'])
    np.random.seed(config['general']['seed'])
    random.seed(config['general']['seed'])
    torch.cuda.manual_seed(config['general']['seed'])

    #protocoll and name the run
    date = datetime.now()
    run_name = config['name'] + '_' + date.strftime(format='str')
    print(run_name)

    #create output directory
    #importet from the project FFTRadNet by Valeo #
    #############################################################################################################################
    ##output_folder = Path("C:/Users/mail/OneDrive/Dokumente/ImRad/output")
    output_folder = Path("C:/Users/mail/OneDrive - bwedu/Semester 5 (6)/Entwicklungsprojekt/Clone_29_04/ImageRadar/output")
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / run_name).mkdir(parents=True, exist_ok=True)
    #############################################################################################################################

    #set device -> CPU or GPU
    def get_device():
    # check if GPU is availabale, else use CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device
    device = get_device()


    # Initiate Tensorboard for visualization of processes 
    #log_write = SW(output_folder)

    #load dataset
    #########
    ## dataset = ImRad_PCL(root_dir = 'C:/Users/mail/OneDrive/Dokumente/ImRad/radar_PCL')
    dataset = ImRad_PCL(root_dir = 'C:/Users/mail/OneDrive - bwedu/Semester 5 (6)/Entwicklungsprojekt/Clone_29_04/ImageRadar/radar_PCL')
    #df, box_labels, dataset_RPC = data_pcl.CreateDataset()
    #########   
    ######################
    ######################
    train_loader,test_loader = data_pcl.CreateDataLoaders(dataset)
    #train_loader, test_loader = CreateDataLoaders(dataset,config['dataloader'],config['seed']
    #create model
    net = ImRadNet()

    Norm_Data = DataNorm()
    
    #load model to selected device
    net.to(device)

    # Optimizer
    lr = float(config['optimizer']['lr'])                           # define initial learning rate of first iteration
    step_size = int(config['learning_rate']['step_size'])           # defines how many epochs should be run without changing the learning rate
    gamma = float(config['learning_rate']['gamma'])                 # after the number of epochs defined in step_size, the learning rate is multiplied (in this case reduced) with the factor gamma
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    num_epochs=int(config['num_epochs'])    # number of total epochs defined in config

    # Start Training
    start_epoch = 0
    counter = 0
    #freespace_loss = nn.BCEWithLogitsLoss(reduction='mean')     ########################   
    freespace_loss = nn.L1Loss()

    '''
    if resume:
        print('===========  Resume training  ==================:')
        dict = torch.load(resume)
        net.load_state_dict(dict['net_state_dict'])
        optimizer.load_state_dict(dict['optimizer'])
        scheduler.load_state_dict(dict['scheduler'])
        startEpoch = dict['epoch']+1
        history = dict['history']
        global_step = dict['global_step']

        print('       ... Start at epoch:',startEpoch)
    '''
    for epoch in range(start_epoch,num_epochs):
        net.train()
        running_loss = 0.0  ########################
        loss_summed = 0
        print('Epoch #',epoch)


        # for i,data in enumerate(train_loader.dataset.dataset):
        #     for j,labels in enumerate(train_loader.dataset.indices):
        for i,(data,labels) in enumerate(zip(train_loader.dataset.dataset, train_loader.dataset.indices)):
            input = torch.Tensor([data[0:3]]).to(device).float()
            inputs = Norm_Data(input)
            #label_map = torch.Tensor(data[0]).to(device).float()
            # if 'model' in config:
            #     if 'SegmentationHead' in config['model']:
            #         seg_map_label = torch.Tensor(data[2]).to(device).double()
            # else:
            seg_map_label = torch.Tensor(labels[0]).to(device).float()
            '''
            if(config['model']['SegmentationHead']=='True'):
            seg_map_label = data[2].to(device).double()
            '''
            # reset the gradient
            optimizer.zero_grad()

            # forward pass, enable to track our gradient
            with torch.set_grad_enabled(True):
                outputs = net(inputs)
            
                            
            prediction = outputs.contiguous().flatten()
            label = seg_map_label.contiguous().flatten()  
            loss = freespace_loss(prediction,label)
            #loss *= inputs.size(0)

            # calculate losses
            #classif_loss, reg_loss = pixor_loss(outputs, label_map,config['losses'])           

            #classif_loss *= config['losses']['weight'][0]
            #reg_loss *= config['losses']['weight'][1]
            #loss_seg *=config['losses']['weight'][2]
            loss *= config['losses']['weight'][2]

            ## calculate total loss
            #loss =  loss_seg ## +classif_loss +reg_loss
            #print(classif_loss)
            #print(reg_loss)
            #print(loss_seg)
            #print(loss)
            
            # backpropagation
            loss.backward()
            optimizer.step()

            # statistics
            # running_loss += loss.item() * inputs.size(0) 

            loss_summed += loss
            counter += 1
        print(loss_summed/2500)
        print("counter: ", counter)

        scheduler.step()

        '''
        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['lr'].append(scheduler.get_last_lr()[0])
        '''
        ###################################################
        # validation phase was cut out due to simplicity  #
        ###################################################  

    torch.save({
        'model_state_dict':net.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        
    },'ImRad.pth')
    print('')


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='ImRadNet Training')
    ## parser.add_argument('-c', '--config', default='config/config.json',type=str,
    ##                    help='Path to the config file (default: config.json)')
    parser.add_argument('-c', '--config', default='C:/Users/mail/OneDrive - bwedu/Semester 5 (6)/Entwicklungsprojekt/Clone_29_04/ImageRadar/config/config.json',type=str,
                    help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')

    args = parser.parse_args()

    config = json.load(open(args.config))
    
    main(config)