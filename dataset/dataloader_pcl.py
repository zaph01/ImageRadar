#!/usr/bin/env python
# coding: utf-8

# # Datenloader-File

# In[2]:


import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch

import os
import pandas as pd

from torchvision.transforms import Resize,CenterCrop


#  Daten Laden in ein Dataset (pandas)

# In[3]:


base_dir = 'E:\\Jakob\\Entwicklungsprojekt\\ungezippt\\RADIal-001\\RADIal'
labels = pd.read_csv(os.path.join(base_dir,'labels.csv'))
labels_np = labels.to_numpy()

# um nur einfache Werte difficult = 1 anzuzeigen
ids_filters=[]
ids = np.where(labels_np[:, -1] == 0)[0]
ids_filters.append(ids)
ids_filters = np.unique(np.concatenate(ids_filters))
labels_np = labels_np[ids_filters]

# um nur noch eindeutige numSamples in 1. Spalte zu haben 
unique_ids = np.unique(labels_np[:,0])
label_dict = {}
for i,ids in enumerate(unique_ids):
    sample_ids = np.where(labels_np[:,0]==ids)[0]
    label_dict[ids]=sample_ids
sample_keys = list(label_dict.keys())

len_dict = len(label_dict)

dataset_RPC = []
box_labels = []
for i in range(3000):
    sample_id = sample_keys[i] 
    # From the sample id, retrieve all the labels ids
    entries_indexes = label_dict[sample_id]
    # Get the objects labels
    box_labels_one = labels_np[entries_indexes]

    # Labels contains following parameters:
    # numSample	x1_pix	y1_pix	x2_pix	y2_pix	laser_X_m	laser_Y_m	laser_Z_m	radar_X_m	radar_Y_m	radar_R_m
    # radar_A_deg	radar_D_mps	radar_P_db
    box_labels_one = box_labels_one[:,10:13].astype(np.float32)
    box_labels.append(box_labels_one)  # Ergebnisse zur Liste hinzufügen
    RPC_filename = os.path.join(base_dir,'radar_PCL',"pcl_{:06d}.npy".format(sample_id))
    # range,azimuth,elevation,power,doppler,x,y,z,v  ---> [0,1,4]
    dataset_RPC.append(np.load(RPC_filename,allow_pickle=True)[[0,1,4],:])
    
    df = pd.DataFrame({'Data_RPC': dataset_RPC, 'Labels': box_labels})


# Klasse zum erstellen von Batch 

# In[4]:


def RADIal_collate(batch):
    radar_pcs = []
    labels = []

    for dataset_RPC, box_labels in batch:
        labels.append(torch.from_numpy(box_labels))           # torch.from_numpy  -> gibt einen Tensor zurück (das box_labels array und der
                                                              # und der entstandene Tensor teilen sich einen Speicher, größe des Tensors lässt sich ncicht ändern)
        radar_pcs.append(torch.from_numpy(dataset_RPC))
        
    return radar_pcs,labels            # torch.stack  -> verkettet Tensoren entlang einer neuen Dimension 


# Klasse zur Loader Erstellung 

# In[5]:


def CreateDataLoaders(dataset,batch_size=4,shuffle=True,num_workers=0,seed=0):   # batch_size -> Datenloader lädt Daten in Batches mit angegebener Größe
                                                                    # num_workers -> Anzahl Prozesse, die zum Laden der Daten verwendet werden (Ladegescheindigkeit kann erhöht werden)
                                                                    # seed -> wichtig für Reproduzierbarkeit der Ergebnisse 
                                                                    # shuffle -> zum Durchmische des datasets
    dict_index_to_keys = {s:i for i,s in enumerate(sample_keys)}    # jedem dataset.sample_kexs wird ein index zugeordnet 
    
    Test_indexes = []                                 # Daten werden aufgeteilt 
    for i in range(500):                              # durchlaufen der Daten 
        Test_indexes.append(box_labels[i])            # index der Sequenz wird gefunden und Liste hinzugefügt 
   
    Train_indexes = []
    for i in range(500, 2500):
        Train_indexes.append(box_labels[i])

                                                                                         # gibt die werte von array 1 an, die nicht in array 2 vorkommen 
    train_dataset = Subset(dataset,Train_indexes)                  
    test_dataset = Subset(dataset,Test_indexes)

    # Erstellen der data_loaders (um die Daten in Batches zu laden, zu mischen und zu transformieren, bevor sie in das Modell eingespeist werden)
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=RADIal_collate)

    test_loader =  DataLoader(test_dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=RADIal_collate)
 
    return train_loader,test_loader


# aufruf der Loader-Datei

# In[6]:


train_loader, test_loader = CreateDataLoaders(df)


# In[ ]:




