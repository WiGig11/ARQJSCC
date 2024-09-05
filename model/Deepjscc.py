import torch
import torch.nn as nn
from torch.optim import Adam

from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

import torchvision
from torchvision import models

import pytorch_lightning as pl
from pytorch_lightning  import LightningModule

import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pdb
import time



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.show()

class DeepJSCCARQ(LightningModule):
    def __init__(self,encoder,decoder,loss_module_D,loss_module_G,channel,discriminator,hyperparameter,lr_scheduler_type,lr_D,lr_G):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_module_D = loss_module_D#compute loss 
        self.loss_module_G = loss_module_G
        self.channel = channel#simulate channel
        self.discriminator = discriminator
        self.hyperparameter = hyperparameter
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_D = lr_D
        self.lr_G = lr_G
        self.automatic_optimization = False
        self.generator_params = list(self.encoder.parameters()) + list(self.decoder.parameters())

    def forward(self,image,snr):
        encoded = self.encoder.forward(image,snr)
        received = self.channel.forward(encoded,snr)
        decoded = self.decoder.forward(received,snr)
        #realout,fakeout = self.discriminator(decoded,image)
        return decoded,received,encoded
        
    def training_step(self,batch):
        snr = random.randint(0, 20)
        image,_ = batch
        image_c = image.clone()
        g_opt, d_opt = self.optimizers()
        sch1,sch2 = self.lr_schedulers()
        decoded,_,_ = self.forward(image=image,snr=snr)
        real_out = self.discriminator(image)
        fake_out= self.discriminator(decoded)
        loss_G,_ = self.loss_module_D(real_out,fake_out)
        loss_G_new = loss_G+self.hyperparameter*self.loss_module_G(image,decoded)
        #opt1 = g_opt
        g_opt.zero_grad()
        self.manual_backward(loss_G_new,retain_graph=True)
        g_opt.step()
        if self.trainer.is_last_batch and (self.trainer.current_epoch+1)%1==0:
            sch2.step()

        decoded_c,_,_ = self.forward(image=image_c,snr=snr)
        real_out_c = self.discriminator(image_c)
        fake_out_C = self.discriminator(decoded_c)
        _,loss_D_c = self.loss_module_D(real_out_c,fake_out_C)
        #opt = d_opt
        d_opt.zero_grad()
        self.manual_backward(loss_D_c)
        d_opt.step()
        if self.trainer.is_last_batch and (self.trainer.current_epoch+1)%1==0:
            sch1.step()
            
        del image_c,decoded_c,real_out_c,fake_out_C
        loss_G = loss_G_new.to(self.device)
        loss_D = loss_D_c.to(self.device)
        loss_G_mse = self.hyperparameter*self.loss_module_G(image,decoded).to(self.device)
        self.log('training loss G',loss_G,on_step = False, on_epoch = True,prog_bar = True,logger = True, sync_dist=True)
        self.log('training loss G_mse',loss_G_mse,on_step = False, on_epoch = True,prog_bar = True,logger = True, sync_dist=True)
        self.log('training loss D',loss_D,on_step = False, on_epoch = True,prog_bar = True,logger = True, sync_dist=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning rate',current_lr,on_step = False, on_epoch = True,prog_bar = True,logger = True, sync_dist=True)

    def validation_step(self,batch,batch_idx):
        snr = int(15)
        source_image,_ = batch
        decoded,_,_ = self.forward(source_image,snr)
        real_out = self.discriminator(source_image)
        fake_out = self.discriminator(decoded)
        loss_G,loss_D = self.loss_module_D(real_out,fake_out)
        loss_G_new = loss_G+self.loss_module_G(source_image,decoded)
        loss_D = loss_D.to(self.device)
        loss_G = loss_G_new.to(self.device)
        example = decoded[0]
        mse = torch.mean(((source_image/2+0.5) - (example/2+0.5)) ** 2, dim=[1, 2, 3])
        psnr = 10*torch.log10(1/mse)
        self.logger.experiment.add_image('source',source_image[0]/2+0.5,self.current_epoch)
        #self.logger.experiment.add_image('target',target_image[0],self.current_epoch)
        self.logger.experiment.add_image('val_res',example/2+0.5,self.current_epoch)
        self.log('psnr',torch.mean(psnr),on_step = False, on_epoch = True,prog_bar = True,logger = True, sync_dist=True)
        self.log('val_loss_D',loss_D,on_step = False, on_epoch = True,prog_bar = True,logger = True, sync_dist=True)
        self.log('val_loss_G',loss_G,on_step = False, on_epoch = True,prog_bar = True,logger = True, sync_dist=True)
        val_loss = loss_D+loss_G_new
        self.log('val_loss',val_loss,on_step = False, on_epoch = True,prog_bar = True,logger = True, sync_dist=True)
        #return loss_D,loss_G

        
    def configure_optimizers(self):
        optimizer_G = Adam(self.generator_params, lr=self.lr_G)
        optimizer_D = Adam(self.discriminator.parameters(), lr=self.lr_D)
        lr_scheduler_type = self.lr_scheduler_type
        if 'step' in lr_scheduler_type.lower():
            #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones = [500,1000,1500],gamma = 0.1)
            #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones = [500,1000,1500],gamma = 0.1)
            scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G,step_size = 300,gamma = 0.1)
            scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D,step_size = 300,gamma = 0.1)
        else:
            pass
            #scheduler = {"scheduler":torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = 10),
            #            "monitor":"val loss"}
        return ({"optimizer": optimizer_G, "lr_scheduler": scheduler_G},
        {"optimizer": optimizer_D, "lr_scheduler": scheduler_D})