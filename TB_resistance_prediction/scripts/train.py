import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score, average_precision_score, precision_recall_curve
import wandb
import numpy as np

class TrainingModule(LightningModule):
    def __init__(
        self,
        model,
        lr,
        record_run
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.record_run = record_run
        self.loss_function = nn.BCELoss()
        self.validation_step_aucs = []
        self.test_step_aucs = []

        if self.record_run:
        # initialize weigth&biases run 
            wandb.init(
                project='resistance_prediction_test_2',
                name = 'ALL_train_strains_and_test_wand_ALL',
                config={
                    'learning_rate': 3e-4,
                    'architecture': 'CNN',
                    'dataset': 'batch',
                    'epochs': 4
                }
            )


    def forward(self, batch):

        # check if there is a CUDA-enable GPU available for use 
        if torch.cuda.is_available():
            batch = batch.cuda()

        output = self.model(batch)
        return output

    def loss(self, output, labels):
        return self.loss_function(output, labels)

    def step(self, batch):
        data = batch['features']
        labels = batch['labels']

        #import pdb
        #pdb.set_trace()

        forward_out = self.forward(data)
        loss = self.loss(forward_out.to(torch.float), labels.to(torch.float))

        # need to convert tensors that are located on the CUDA device  to NumPy arrays directly
        labels_cpu = labels.to(torch.float).cpu()
        forward_out_cpu = forward_out.to(torch.float).cpu()

        print('PREDICTIONS : ', forward_out_cpu.detach().numpy())
        print('LABELS : ', labels_cpu.detach().numpy())

        # compute false positive rate, true positive rate and AUC
        precision, recall, _ = precision_recall_curve(labels_cpu.detach().numpy(), forward_out_cpu.detach().numpy())
        roc_auc = auc(recall, precision)


        return loss, roc_auc

    def training_step(self, batch):
        loss, roc_auc = self.step(batch)

        # store wandb
        if self.record_run : 
            wandb.log({'loss_training': loss})
            wandb.log({'AUC_training': roc_auc})

        return loss

    def validation_step(self, batch, batch_idx):
        loss, roc_auc = self.step(batch)

        self.log('val_auc', roc_auc, on_epoch=True)
        self.validation_step_aucs.append(roc_auc)

        return loss

    
    def on_validation_epoch_end(self) : 
        '''
        AUC mean computation of all AUCs in one epoch 
        function called at the end of each epoch 
        enables to log in wandb validation AUC for each epoch
        '''
        #print('AUC END EPOCH ', np.mean(self.validation_step_aucs))
                
        if self.record_run : 
            wandb.log({'AUC_validation': np.mean(self.validation_step_aucs)})

        self.validation_step_aucs.clear()



    def test_step(self, batch, batch_idx):
        loss, roc_auc = self.step(batch)
        print('TEST LOSS : ', loss)

        self.test_step_aucs.append(roc_auc)

        # store wandb
        #if self.record_run : 
            #wandb.log({'loss_test': loss})
            #wandb.log({'AUC_test': roc_auc})

        return loss
    
    #def on_test_epoch_end(self): 
        #print('ENDDD TESTTTTT')


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer



