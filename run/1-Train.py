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
from model.
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

    #set device
    device = get_device()

    
    #load dataset



    #create model
    net = ImRadNet()

