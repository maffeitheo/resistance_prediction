from pytorch_lightning.callbacks import Callback
import numpy as np
import wandb


class TestCallback(Callback): 

    def __init__(self, record_run): 
        self.record_run = record_run


    def on_test_epoch_end(self, trainer, pl_module): 
        '''
        function called at the end of all epochs of the test
        stores into wandb test AUC (computed as the mean of all AUCs of each epoch of the test)
        '''
        if self.record_run : 
            wandb.log({'AUC_test': np.mean(pl_module.test_step_aucs)})



