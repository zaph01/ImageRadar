############################### IMPORTANT INFORMATION ###################################
# This code contains functions from RADIal-Repository by Valeo.ai                       #  
# For further information go visit https://github.com/valeoai/RADIal?tab=readme-ov-file #
#########################################################################################
 
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import os
import pandas as pd
 
class ImRad_PCL(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.labels = pd.read_csv(os.path.join(self.root_dir,'labels.csv'))
        self.labels_np = self.labels.to_numpy()
 
        # only get simple values difficult = 0
        ids_filters=[]
        ids = np.where(self.labels_np[:, -1] == 0)[0]
        ids_filters.append(ids)
        ids_filters = np.unique(np.concatenate(ids_filters))
        self.labels_np = self.labels_np[ids_filters]
        self.unique_ids = np.unique(self.labels_np[:,0])
        self.label_dict = {}
        for i,ids in enumerate(self.unique_ids):
            sample_ids = np.where(self.labels_np[:,0]==ids)[0]
            self.label_dict[ids]=sample_ids
        self.sample_keys = list(self.label_dict.keys())
   
    def __len__(self):
        return len(self.label_dict)
 
    def __getitem__(self,index):
        dataset_RPC = []
        box_labels = []
        sample_id = self.sample_keys[index]
        # From the sample id, retrieve all the labels ids
        entries_indexes = self.label_dict[sample_id]
        # Get the objects labels
        box_labels = self.labels_np[entries_indexes]
 
        # Labels contains following parameters:
        # numSample x1_pix  y1_pix  x2_pix  y2_pix  laser_X_m   laser_Y_m   laser_Z_m   radar_X_m   radar_Y_m   radar_R_m
        # radar_A_deg   radar_D_mps radar_P_db
        box_labels = box_labels[:,10:13].astype(np.float32)
        #box_labels.append(box_labels)  # Ergebnisse zur Liste hinzufügen
        RPC_filename = os.path.join(self.root_dir,"pcl_{:06d}.npy".format(sample_id))
        # range,azimuth,elevation,power,doppler,x,y,z,v  ---> [0,1,4]
        dataset_RPC.append(np.load(RPC_filename,allow_pickle=True)[[0,1,4],:])
        return dataset_RPC, box_labels
    # um nur noch eindeutige numSamples in 1. Spalte zu haben
   
# Class to create batches
def ImRad_collate(batch):
    #df, box_labels, dataset_RPC = CreateDataset()
    #dataset_RPC, box_labels = ImRad_PCL(root_dir='C:/Users/mail/OneDrive/Dokumente/ImRad/radar_PCL')
    radar_pcs = []
    labels = []
 
    for dataset_RPC, box_labels in batch:
        labels.append(torch.from_numpy(box_labels))           # torch.from_numpy  -> gibt einen Tensor zurück (das box_labels array und der
                                                              # und der entstandene Tensor teilen sich einen Speicher, größe des Tensors lässt sich ncicht ändern)
        radar_pcs.append(torch.from_numpy(dataset_RPC))
       
    return torch.stack(radar_pcs),labels          # torch.stack  -> verkettet Tensoren entlang einer neuen Dimension
 
 
# Class for loader settings
 
def CreateDataLoaders(dataset,batch_size=4,shuffle=True,num_workers=2,seed=3):   # batch_size -> Datenloader loads data in batches
                                                                                 # num_workers -> Number of processes used to load the data (loading speed can be increased)
    data = []
    box_labels_append = []
    for i in range(dataset.__len__()):
        #data = dataset.__getitem__(i)[0]
        data.append(dataset.__getitem__(i)[0])
        #box_labels= dataset.__getitem__(i)[1]
        box_labels_append.append(dataset.__getitem__(i)[1])
                                                                      # seed -> Important for reproducibility of results
 
    Test_indexes = []                                                 # seperate data
    test_data = []
    for i in range(500):                                              # run through data
        test_data.append(data[i])
        Test_indexes.append(box_labels_append[i])                     # find index and append to list
 
    Train_indexes = []
    train_data = []    
    for i in range(500, 3000):
        train_data.append(data[i])
        Train_indexes.append(box_labels_append[i])
 
                                                                                         # specifies the values of array 1 that do not occur in array 2
    train_dataset = Subset(train_data,Train_indexes)                  
    test_dataset = Subset(test_data,Test_indexes)
 
    # Create the data_loaders (to load, merge and transform the data in batches before feeding it into the model)
    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=ImRad_collate)
 
    test_loader =  DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=ImRad_collate)
 
    return train_loader,test_loader