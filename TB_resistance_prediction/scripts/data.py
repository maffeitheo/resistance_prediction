#from pytorchlightning import LightningDataModule
#import pytorch_lightning
#from torch.utils.data import random_split
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

        

class NucleotideDataSet(LightningDataModule):

    def __init__(self, RIF_sequence_path, train_strains_path, test_strains_path, use_dataset_train, batch_size, simple_sampler=True):
        
        super().__init__()
        
        self.RIF_sequence_path = RIF_sequence_path 
        self.train_strains_path = train_strains_path
        self.test_strains_path = test_strains_path
        self.use_dataset_train = use_dataset_train
        self.batch_size = batch_size
        self.simple_sampler = simple_sampler

        self.sequences = self.read_data_from_file(self.RIF_sequence_path)
        self.train_resistances = self.read_data_from_file(self.train_strains_path)
        self.test_resistances = self.read_data_from_file(self.test_strains_path)

        
    def read_data_from_file(self, data_file):
        '''
        return RIF sequences and train strains data into a list
        '''
        data = []
        with open(data_file, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    ID, gene, sequence = parts
                    data.append((ID, gene, sequence))
                if len(parts) == 2:
                    ID, resistance = parts
                    data.append((ID, resistance))
        return data
    
    def __len__(self):
        '''
        return length of train of test strains
        '''
        if self.use_dataset_train : return len(self.read_data_from_file(self.train_strains_path))
        else : len(self.read_data_from_file(self.test_strains_path))

    def __getitem__(self, i):
        '''
        parameter
        a) if simple_sampler True (from class SimpleSampler) : i is the index ID 
        b) if not simple_sampler : i is the i-th element (i is a number)
        return gene ID, nucleotide sequences associated to gene ID and resistance value separately for train and test dataset
        '''

        # when you use the class SimpleSampler to randomly sample dataset (in order to have balanced data)
        if self.simple_sampler :

            sequences = []
            resistance = []
            ID = []
            longest_length = 7550
            
            ## TRAIN
            if self.use_dataset_train == True:
                #print('TRAIN RESISTANCE ', self.train_resistances)

                for line in self.train_resistances : 
                    if line[0] == i : 
                        ID, resistance = line
                        print('LINE ', line)

                for line in self.sequences : 
                    if line[0] == ID : 
                        # pad the sequences to the maximal nucleotide length
                        padding_needed = longest_length - self.one_hot_encode(line[2]).shape[0]
                        padded_sequence = np.vstack((self.one_hot_encode(line[2]), np.zeros((padding_needed, 5))))
                        sequences.append(padded_sequence)
            ## TEST
            else : 
                print('TEST RESISTANCE  III ', i)

                for line in self.test_resistances : 
                    if line[0] == i : 
                        ID, resistance = line
                        print('LINE ', line)

                for line in self.sequences : 
                    if line[0] == ID : 
                        # pad the sequences to the maximal nucleotide length
                        padding_needed = longest_length - self.one_hot_encode(line[2]).shape[0]
                        padded_sequence = np.vstack((self.one_hot_encode(line[2]), np.zeros((padding_needed, 5))))
                        sequences.append(padded_sequence)
            return {
                'ID': ID,
                'sequences': sequences, 
                'resistance' : resistance}
        
        # when you do not perform balancing, i is the i-th element (a number)
        else :
            sequences = []
            resistance = []
            ID = []
            longest_length = 7550
            
            ## TRAIN
            if self.use_dataset_train == True:
                print('TRAIN RESISTANE ', self.train_resistances)
                ID, resistance = self.train_resistances[i]
            
                for line in self.sequences : 
                    if line[0] == ID : 
                        # pad the sequences to the maximal nucleotide length
                        padding_needed = longest_length - self.one_hot_encode(line[2]).shape[0]
                        padded_sequence = np.vstack((self.one_hot_encode(line[2]), np.zeros((padding_needed, 5))))
                        sequences.append(padded_sequence)
            ## TEST
            else : 
                ID, resistance = self.test_resistances[i]
            
                for line in self.sequences : 
                    if line[0] == ID : 
                        # pad the sequences to the maximal nucleotide length
                        padding_needed = longest_length - self.one_hot_encode(line[2]).shape[0]
                        padded_sequence = np.vstack((self.one_hot_encode(line[2]), np.zeros((padding_needed, 5))))
                        sequences.append(padded_sequence)
                
            return {
                'ID': ID,
                'sequences': sequences, 
                'resistance' : resistance}
    
    def one_hot_encode(self, sequence):
        '''
        return array of array, one array is one nucleotide
        if the sequence is None, return array([[0., 0., 0., 0., 0.]])
        '''
        try :
            mapping = dict(zip("ACGTX", range(5)))
            if sequence:
                seq2 = [mapping[i] for i in sequence]
            else:
                seq2 = []
            return (np.eye(5)[seq2])
        except :
            return np.array([[0., 0., 0., 0., 0.]])
        
        
    def collate_fn(self, batch): 

        X = []
        Y = []

        for i in batch : 
            X.append(i['sequences'])
            #print('I RESISTACE', i['resistance'])
            Y.append([int(i['resistance'])])

        final_batch = {}
        final_batch['features'] = torch.FloatTensor(X)
        final_batch['labels'] = torch.as_tensor(Y)
        #final_batch['labels'] = [torch.as_tensor(Y) for item in Y]
        #print(final_batch)

        return final_batch
        
    # different return based on the presence or not of SimpleSampler (in order to balance data)
    def train_dataloader(self, sampler, use_sampler=True):
        '''
        params : sampler (Sampler from torch.utils.data)
        return DataLoader for train 
        '''
        if use_sampler : 
            batch_sampler = torch.utils.data.sampler.BatchSampler(
                sampler,
                batch_size = self.batch_size,
                drop_last = False
            )
            return DataLoader(self, num_workers = 0, collate_fn=self.collate_fn, batch_sampler=batch_sampler)
        else : return DataLoader(self, batch_size=self.batch_size, num_workers = 0, collate_fn=self.collate_fn)
                        

    def val_dataloader(self, sampler, use_sampler=True):
        '''
        params : sampler (Sampler from torch.utils.data)
        return DataLoader for validation 
        '''
        if use_sampler : 
            batch_sampler = torch.utils.data.sampler.BatchSampler(
                    sampler,
                    batch_size = self.batch_size,
                    drop_last = False
                )
            return DataLoader(self, num_workers = 0, collate_fn=self.collate_fn, batch_sampler=batch_sampler)
        else : return DataLoader(self, batch_size=self.batch_size, num_workers = 0, collate_fn=self.collate_fn)


    def test_dataloader(self):
        return DataLoader(self, num_workers=0, batch_size=self.batch_size, collate_fn=self.collate_fn)














