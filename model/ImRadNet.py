import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from torchvision.transforms.transforms import Sequence

# Set global constants for input tensor
BATCH = 4
INPUT_CHANNELS = 3
HIGHT  = 3
WIDTH = 1028

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


def conv3x3_RAD(input, output, stride = 1, bias = False):
    return(nn.Conv3d(input,output,stride=stride, padding=1, padding_mode='zeros',bias=bias))

#Nomalize Input to equal dimensions
 #Dimensions of input: batch_size = 4, Channels = 3,
'''
class DataNorm(nn.utils.dataset):
    def __init__(self,input_data, mode = 'bilinear',size = (1028)):     # Size variabel, 1028 nur als vorläufiger Platzhalter
        super(DataNorm,self).__init__()
        self.mode = mode
        self.size = size
        self.input_data = input_data
        self.input_tensor = torch.Tensor(input_data)     # Tensor in 2D -> Shape of input Array -> 3xY: (Line 1: Range, Line 2: Azimuth, Line 3: Doppler)


    def DimNorm(self):    #input dimensions: mini-batch x channels x [optional depth] x [optional height] x width.
        #Normalize Data using interpolation
        tensor_scaled = F.interpolate(input = self.input_tensor, size = self.size, mode = self.mode)    # scaled Tensor: Dimension 2D -> 3 Lines(Order described above), 1028 columns(values interpolated)
        num_columns = tensor_scaled.size(1)     # extract number of columns -> should be equal to self.size

        if self.size != num_columns:
            print("Error in calculating Tensor Dimensions")
            os.abort
        
        return tensor_scaled
'''



class ImRadNet(nn.Module):
    def __init__(self,kernel_size = 3, stride = 1, padding = 1, padding_mode = 'zeros', bias = False):
        super(ImRadNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,padding_mode=padding_mode)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,padding_mode=padding_mode)
        self.fc1 = nn.Linear(96,1028)
        self.fc2 = nn.Linear(1028,128)
        self.fc3 = nn.Linear(128,3)


    def forward(self,x): 
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1,3*32) 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    