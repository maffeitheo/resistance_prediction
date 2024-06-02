import torch
from torch.utils.data import Sampler
import numpy as np
from sklearn.model_selection import train_test_split
#from utils.general_utility_functions import *
#from utils.constants import *
#from tqdm import tqdm

def breakdown_file(file_path, stratify = False):
	if stratify:
		pos_strains = []
		neg_strains = []
	else:
		strains = []

	for line in open(file_path, 'r').readlines():
		data = line.split('\t')
		strain = data[0]
		label = float(data[1])

		if strain != 'None':
			if stratify:
				if label:
					pos_strains.append(strain)
				else:
					neg_strains.append(strain)
			else:
				strains.append(strain)

	if stratify:
		return pos_strains, neg_strains
	else:
		np.random.shuffle(strains)
		return strains


class TestSampler (Sampler[int]) :

    def __init__(self, test_file_path): 
        self.test_strains = breakdown_file(test_file_path, False)

        self.test_strains = np.array(self.test_strains)

        print('STRAINS ', type(self.test_strains))

        #self.test_strains = np.array(self.test_strains)
        self.count = 0

    def breakdown_file(self, file_path): 
        '''
        return train strain ID into a list
        '''
        test_strains = []
        print('FILE PATH ', file_path)

        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')

                if len(parts) == 2:
                    ID, resistance = parts
                    test_strains.append(ID)

            return test_strains
        
    def __len__(self):
        return len(self.test_strains)
    
    def __iter__(self):
        while True:
            yield self.test_strains[self.count % len(self.test_strains)]
            self.count += 1




class SimpleSampler(Sampler[int]) : 

    def __init__(self, train_file_path, test_file_path): 
        self.pos_train_strains, self.neg_train_strains = self.breakdown_file(train_file_path)
        self.pos_test_strains, self.neg_test_strains = self.breakdown_file(test_file_path)

        self.pos_train_strains = np.array(self.pos_train_strains)
        self.neg_train_strains = np.array(self.neg_train_strains)

    def breakdown_file(self, file_path): 
        '''
        return resistant and non resistant strain ID into a list
        '''
        pos_strains = []
        neg_strains = []
        print('FILE PATH ', file_path)

        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')

                if len(parts) == 2:
                    ID, resistance = parts

                    if float(resistance):
                        #print('RESISTANT ', ID)
                        pos_strains.append(ID)
                    else : 
                        #print('NON RESISTANT ', ID)
                        neg_strains.append(ID)

            return pos_strains, neg_strains


    def __len__(self):
         return len(self.pos_train_strains) + len(self.neg_train_strains)
    

    def get_train_sample_num(self, train_sample_size):
        '''
        return 
        '''
        strains_to_train = []
        pos = True
        
        for i in range(train_sample_size):
            if pos:
                strains_to_train.append(np.random.choice(self.pos_train_strains, 1)[0])
                pos = False
            else:
                strains_to_train.append(np.random.choice(self.neg_train_strains, 1)[0])
                pos = True
                return strains_to_train


    def __iter__(self):
        '''
        yield alternation of resistant and non resistant IDs (based on resistant and non resistant strains) 
        '''


        pos = True
        while True:
            if pos:
                #print(self.pos_test_strains)
                strain = np.random.choice(self.pos_train_strains, 1)[0]
                pos = False
            else:
                #print(self.neg_test_strains)
                strain = np.random.choice(self.neg_train_strains, 1)[0]
                pos = True

            yield strain
                        







