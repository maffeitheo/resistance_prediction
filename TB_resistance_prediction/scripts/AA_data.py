from torch.utils.data import Dataset
import esm
import torch
import numpy as np

class AminoAcidDataset(Dataset) : 

    def __init__(self, aa_RIF_sequence_path, train_strains_path, test_strains_path, use_dataset_train): 
        self.aa_RIF_sequence_path = aa_RIF_sequence_path
        self.train_strains_path = train_strains_path
        self.test_strains_path = test_strains_path
        self.use_dataset_train = use_dataset_train

        self.aa_sequences = self.read_data_from_file(self.aa_RIF_sequence_path)
        self.train_resistances = self.read_data_from_file(self.train_strains_path)
        self.test_resistances = self.read_data_from_file(self.test_strains_path)


    def read_data_from_file(self, data_file):
        '''
        return RIF sequences and train strains data into a list
        '''
        data = []
        with open(data_file, 'r') as file:
            for line in file:
                parts = line.split()
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


    def __getitem__(self, idx): 
    
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

            string_sequences = []
            resistance = []
            ID = []
            tokens = []
            # the longest AA sequence has length int(7550/3)+1, nevertheless maximal size could be 512 due to trasnformer constraint
            longest_length_aa_sequence = 512
            tokens_array_padded = []
            
            ## TRAIN
            if self.use_dataset_train:

                ID, resistance = self.train_resistances[idx]

                for line in self.aa_sequences : 
                    if line[0] == ID : 
                        print('LINE ID : ', line[0])
                        # if the dimension exceed memory capability, need to perform embedding if AA sequence larger than 512 aa
                        if line[2] == 'None' : 
                            print('LINE NONE ', line[2])
                            string_sequences.append('None')
                            tokens_array_padded.append(np.ones(512, dtype=int))
                        else : 
                            if len(line[2]) < 512 : 
                                print('LINE < 512 ', line[2])
                                string_sequences.append(line[2])
                                tokens_array_padded.append(alphabet.encode(line[2]) + [1] * (longest_length_aa_sequence - len(alphabet.encode(line[2]))))
                            else : 
                                print('LINE > 512 ', len(line[2]))

                                # embedding
                                original_sequence = alphabet.encode(line[2])
                                segment_size = len(original_sequence) // longest_length_aa_sequence
                                embedded_sequence = []
                                for i in range(longest_length_aa_sequence):
                                    segment_start = i * segment_size
                                    segment_end = (i + 1) * segment_size
                                    segment_mean = int(np.mean(original_sequence[segment_start:segment_end]))
                                    # ensure the values are between 0 and 20
                                    segment_mean = max(0, min(20, segment_mean))
                                    embedded_sequence.append(segment_mean)

                                string_sequences.append(line[2])
                                tokens_array_padded.append(embedded_sequence)


            #tokens_array_padded = [arr + [0] * (longest_length_aa_sequence - len(arr)) for arr in tokens_array]
            #tokens = [torch.tensor(tok_i, dtype=torch.int64) for tok_i in tokens_array_padded]

            tokens = torch.as_tensor(tokens_array_padded)

            return {
                'ID': ID,
                'sequences': string_sequences, 
                'tokens' : tokens}


RIF_sequence_file = '/home/thm333/TB_resistance_prediction_2/data/aa_RIF.txt'
train_strains_path = '/home/thm333/TB_resistance_prediction_2/data/8_RIF_0.0_MUTATION_SPLIT_0_TRAIN.txt'
test_strains_path = '/home/thm333/TB_resistance_prediction_2/data/RIF_0.0_MUTATION_SPLIT_0_TEST'

#AA_module = AminoAcidDataset(RIF_sequence_file, train_strains_path, test_strains_path, use_dataset_train=True)

#print(AA_module.__getitem__(1)['tokens'])





