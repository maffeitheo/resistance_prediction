import torch
import esm
import sys, os
#sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "data"))
from AA_data import AminoAcidDataset

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results


# EXAMPLE DATA 
data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein3",  "ILERSKEPVSGAQLAEELSVS"),
    ("protein4",  "DLIRD<mask>ISQTGMPPTRAEI"),
    ("protein5",  "TARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKAL"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
print(alphabet.padding_idx)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)


# RIF DATA 
RIF_sequence_file = '/home/thm333/TB_resistance_prediction_2/data/aa_RIF.txt'
train_strains_path = '/home/thm333/TB_resistance_prediction_2/data/RIF_0.0_MUTATION_SPLIT_0_TRAIN'
test_strains_path = '/home/thm333/TB_resistance_prediction_2/data/RIF_0.0_MUTATION_SPLIT_0_TEST'
AA_module = AminoAcidDataset(RIF_sequence_file, train_strains_path, test_strains_path, use_dataset_train=True)


#print('BATCH TOKENS RIF DATA  ', AA_module.__getitem__(1)['tokens'])
#print('BATCH TOKENS FAKE  ', batch_tokens)

raise Exception()
for i in range(0,10): 

    with torch.no_grad():
        #print('BATCH TOKENS ', batch_tokens.shape)
        results = model(AA_module.__getitem__(i)['tokens'], repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]


    batch_lens = (AA_module.__getitem__(i)['tokens'] != alphabet.padding_idx).sum(1)
    sequence_representations = []
    for idx, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[idx, 1 : tokens_len - 1].mean(0))



    print('TOKEN REPRESENTATION SHAPE I ', i,  token_representations)
    print('SEQUENCE REPRESENTATION OF SEQUENCE 0 I ', i, sequence_representations[0])
    print('SEQUENCE REPRESENTATION OF SEQUENCE 1 I ', i, sequence_representations[1])


#print('RESULTS ', results)





