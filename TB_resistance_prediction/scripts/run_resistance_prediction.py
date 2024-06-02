import torch
from cnn_model import CNN  # Import your CNN model
from data import NucleotideDataSet
from train import TrainingModule
import pytorch_lightning
from pytorch_lightning import Trainer
from utils import SimpleSampler, TestSampler
from pytorch_lightning.callbacks import ModelCheckpoint
from callbacks import *

#def get_number_loci(RIF_sequence_path) : 



def main():

    #  hyperparameters
    learning_rate = 3e-4
    batch_size = 64
    record_run_choice = True

    RIF_sequence_file = '/home/thm333/TB_resistance_prediction_2/data/sequences_RIF'
    train_strains = '/home/thm333/TB_resistance_prediction_2/data/RIF_0.0_MUTATION_SPLIT_0_TRAIN'
    test_strains = '/home/thm333/TB_resistance_prediction_2/data/RIF_0.0_MUTATION_SPLIT_0_TEST'
        
    callbacks = []
    

    ## TRAIN

    sampler_params = {
			'train_file_path': train_strains,
			'test_file_path': test_strains
		}

    # Create an instance of the SimpleSampler
    sampler = SimpleSampler(**sampler_params)

    data_module = NucleotideDataSet(RIF_sequence_file, train_strains, test_strains, use_dataset_train=True, batch_size=batch_size)
    data_module_val = NucleotideDataSet(RIF_sequence_file, train_strains, test_strains, use_dataset_train=False, batch_size=batch_size)

    ## CNN wants number of loci and drug
    # 6 is the number of loci for one ID
    cnn_model = CNN(6, 'RIF')  

    # initialize training module
    training_module = TrainingModule(model=cnn_model, lr=learning_rate, record_run=record_run_choice)

    gpus = 1 if torch.cuda.is_available() else 0
    use_gpu = torch.cuda.is_available()
    print('AVAILABLE CUDA ', torch.cuda.is_available())

    callbacks.append(TestCallback(record_run=record_run_choice))

    # run only one batch
    trainer = Trainer(gpus=1,
                      max_epochs=4,
                      callbacks = callbacks, 
                      num_sanity_val_steps=0)
                      #callbacks = [checkpoint_callback])
    
    test_sampler_params = {
			'test_file_path': test_strains
		}
    
    test_sampler = TestSampler(**test_sampler_params)

    print('VAL DATALOADER ', data_module_val.val_dataloader(test_sampler))

    # train model
    trainer.fit(training_module, 
                train_dataloaders = data_module.train_dataloader(sampler),
                val_dataloaders = data_module_val.val_dataloader(test_sampler))



    ## TEST

    test_sampler_params = {
			'test_file_path': test_strains
		}
    
    test_sampler = TestSampler(**test_sampler_params)

    data_module_val = NucleotideDataSet(RIF_sequence_file, train_strains, test_strains, use_dataset_train=False, batch_size=batch_size)
    val_dataloader = data_module_val.val_dataloader(test_sampler)
    print('VAL DATALOADER', val_dataloader)


    trainer.test(dataloaders = val_dataloader, ckpt_path = 'best', verbose = True, datamodule = None)




if __name__ == "__main__":
    main()








