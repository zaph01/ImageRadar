import torch
import torch.nn as nn
import torch.nn.functional as f
import json
import torch.optim as optim
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from torch.optim import lr_scheduler
from model.ImRadNet import ImRadNet
from dataset.dataloader import CreateDataLoaders





def main(config, resume):      
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
    run_name = config['name'] + '_' + date.strftime()
    print(run_name)

    #create output directory
    #importet from the project FFTRadNet by Valeo #
    #############################################################################################################################
    output_folder = Path(config['output']['dir'])
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

    #load dataset

    ######################
    #dataset = 
    ######################
    #train_loader, val_loader, test_loader = CreateDataLoaders(dataset,config['dataloader'],config['seed']
    #create model
    net = ImRadNet()
    
    #load model to selected device
    net.to(device)

    # Optimizer
    lr = float(config['optimizer']['lr'])
    step_size = int(config['learning_rate']['step_size'])
    gamma = float(config['learning_rate']['gamma'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    num_epochs=int(config['num_epochs'])

    # Start Training
    start_epoch = 0
    
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

    for epoch in range(start_epoch,num_epochs):
        net.train()
        #########
        train_loader = 0
        #########

        for i,data in range(train_loader):
            inputs = data[0].to('cuda').float()

            #reset gradient
            optimizer.zero_grad()
             
             
             # forward pass, enable to track our gradient
            with torch.set_grad_enabled(True):
                outputs = net(inputs)
        





