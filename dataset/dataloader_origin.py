# Loader splits dataset in 3 parts: Train, validation and testing
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch

# define dataset for valdaion and testing
Sequences = {'Validation':['RECORD@2020-11-22_12.49.56','RECORD@2020-11-22_12.11.49','RECORD@2020-11-22_12.28.47','RECORD@2020-11-21_14.25.06'],
            'Test':['RECORD@2020-11-22_12.45.05','RECORD@2020-11-22_12.25.47','RECORD@2020-11-22_12.03.47','RECORD@2020-11-22_12.54.38']}
# nach beispiel: ['RECORD@2020-11-22_12.49.56','RECORD@2020-11-22_12.11.49','RECORD@2020-11-22_12.28.47','RECORD@2020-11-21_14.25.06']

# define function to create batches
def RADIal_collate(batch):
    images = []                 # Platzhalter definieren    # create empty arrays
    FFTs = []
    laser_pcs = []
    radar_pcs = []
    segmaps = []
    labels = []

    for image, radar_FFT, radar_pc, laser_pc,segmap,box_labels in batch:
        labels.append(torch.from_numpy(box_labels))           # torch.from_numpy  -> gibt einen Tensor zurück (das box_labels-array
                                                              # und der entstandene Tensor teilen sich einen Speicher, größe des Tensors lässt sich nicht ändern)
        images.append(torch.tensor(image))                    # torch.tensor -> kopiert Daten
        FFTs.append(torch.tensor(radar_FFT).permute(2,0,1))   # torch.permute  -> gibt Teile des Tensor wieder (Postionen werden angegeben)
        segmaps.append(torch.tensor(segmap))
        laser_pcs.append(torch.from_numpy(laser_pc))
        radar_pcs.append(torch.from_numpy(radar_pc))
        
    return torch.stack(images), torch.stack(FFTs), torch.stack(segmaps),laser_pcs,radar_pcs,labels      # torch.stack  -> verkettet Tensoren entlang einer neuen Dimension 

# create data loaders
def CreateDataLoaders(dataset,batch_size=4,num_workers=2,seed=0):   # batch_size -> Datenloader lädt Daten in Batches mit angegebener Größe
                                                                    # num_workers -> Anzahl Prozesse, die zum Laden der Daten verwendet werden (Ladegeschwindigkeit kann erhöht werden)
                                                                    # seed -> wichtig für Reproduzierbarkeit der Ergebnisse 
    dict_index_to_keys = {s:i for i,s in enumerate(dataset.sample_keys)} # jedem dataset.sample_keys wird ein index zugeordnet 

    Val_indexes = []                                           # Daten werden aufgeteilt 
    for seq in Sequences['Validation']:                        # durchlaufen der Daten 
        idx = np.where(dataset.labels[:,14]==seq)[0]
        Val_indexes.append(dataset.labels[idx,0])              # index der Sequenz wird gefunden und Liste hinzugefügt 
    Val_indexes = np.unique(np.concatenate(Val_indexes))       # dopppelte Sequenzen werden rausgenommen 

    Test_indexes = []
    for seq in Sequences['Test']:
        idx = np.where(dataset.labels[:,14]==seq)[0]
        Test_indexes.append(dataset.labels[idx,0])
    Test_indexes = np.unique(np.concatenate(Test_indexes))

    val_ids = [dict_index_to_keys[k] for k in Val_indexes]     # entsprechende Sequenz-IDs werden aus dem Dataset ausgewählt
    test_ids = [dict_index_to_keys[k] for k in Test_indexes]
    train_ids = np.setdiff1d(np.arange(len(dataset)),np.concatenate([val_ids,test_ids]))   # np.setdiff1d -> findet Unterschiede zwischem den beiden arrays
                                                                                           # gibt die werte von array 1 an, die nicht in array 2 vorkommen 
    train_dataset = Subset(dataset,train_ids)                  
    val_dataset = Subset(dataset,val_ids)  
    test_dataset = Subset(dataset,test_ids)

    #Create the data_loaders (to load, merge and transform the data in batches before feeding it into the model)
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=RADIal_collate)
    val_loader =  DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=RADIal_collate)
    test_loader =  DataLoader(test_dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=RADIal_collate)

    return train_loader,val_loader,test_loader