import torch
from torch import nn
from copy import deepcopy
from argparse import ArgumentParser

from datasets.data_loader import get_loaders
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset

import time
import torch
import numpy as np
import numpy

import torch.nn.functional as F

from loggers.exp_logger import ExperimentLogger
from torch.optim import Adam

from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from datasets.exemplars_selection import override_dataset_transform

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision
import random

class Appr(Inc_Learning_Appr):

    def __init__(self, model, device, nepochs=160, lr=0.1, lr_min=1e-4, lr_factor=10, lr_patience=8, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        print("AGLA Incremental Learning")
        
        self.transformed_data=None
        self.batch_size =  None
        
        self.mem_x =  None
        self.mem_y =  None
        
        
        self.outer_epochs = 1
        self.inner_epochs = 1
        

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        return parser.parse_known_args(args)


    def _get_optimizer(self):
        """Returns the optimizer"""
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def _get_optimizer(self, model, lr):
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=self.wd, momentum=self.momentum)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        if (t==0):
            self.batch_size=trn_loader.batch_size
            dims=1
            for images, targets in trn_loader:
                dims = np.array(images[0]).size
                break
            print("Input Size: "+str(dims))
            #self.assessor = LSTMAssessor(dims,self.device).to(self.device)
           


    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        #combine dataset in t>0
        if(t>0):
            self.model.eval()
            self.model.to(self.device)
            prev_aug_outputs = self.model(self.mem_x)
    
           

        trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        self.baseOptimizer = self._get_optimizer(self.model, self.lr)

        #Train inner outer
        for e in range(self.nepochs):

            for oe in range(self.outer_epochs):

                print("[Outer Loop] Train Base ")
                self.model.train()
                clock0 = time.time()
                loss = loss1 = loss2 = 0
                for images, targets in trn_loader:
                    self.model.to(self.device)
                    outputs = self.model(images.to(self.device))

                    if(t>0):
                        curr_aug_outputs = self.model(self.mem_x.to(self.device))
                        loss1 = self.cecriterion(t, outputs, targets.to(self.device))
                        loss2 = self.lderloss(t, prev_aug_outputs, curr_aug_outputs, self.mem_y) 
                        loss = loss1+loss2
                    else:
                        loss = self.cecriterion(t, outputs, targets.to(self.device))

                    self.baseOptimizer.zero_grad()
                    loss.backward()
                    self.baseOptimizer.step()
                    
                #Cek accuracy per epoch
                clock1 = time.time()
                val_loss, val_acc, _ = self.eval(t, val_loader)
                clock2 = time.time()
                print('| O-Epoch {:3d} {:3d}, time={:5.1f}s/{:5.1f}s | lr={:.3f} |Valid: loss={:.3f} {:.3f} {:.3f} {:.3f}, TAw acc={:5.1f}% |'.format(
                e+1, oe+1, clock1 - clock0, clock2 - clock1, self.lr, val_loss, loss, loss1, loss2, 100 * val_acc), end='\n')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=val_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * val_acc, group="train")  

       
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)
        self.precprocess_memory(t, trn_loader, val_loader.dataset.transform)


    def precprocess_memory(self, t, trn_loader, transform):
        isFirst=True
        tmp_x = np.array([])
        tmp_y = np.array([])
        
        with override_dataset_transform(self.exemplars_dataset, transform) as _ds:
            mem_loader = torch.utils.data.DataLoader(_ds, batch_size=trn_loader.batch_size, shuffle=True,
                                                        num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
            for images, targets in mem_loader:
                if(isFirst):
                    isFirst=False
                    tmp_x = images
                    tmp_y = targets
                else:
                    tmp_x = np.concatenate((tmp_x, images),axis=0)
                    tmp_y = np.concatenate((tmp_y, targets),axis=0)

            self.mem_x = torch.tensor(np.array(tmp_x)).type(torch.float).to(self.device)
            self.mem_y = torch.tensor(np.array(tmp_y)).type(torch.float).to(self.device)
            #self.mem_x = self.mem_x.view(self.mem_x.shape[0],self.mem_x.shape[3],self.mem_x.shape[1],self.mem_x.shape[2])
            
    
    def post_train_process(self, t, trn_loader):
        print("Post Train Process")


    def save_aug_image(self, x, fname):
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
        fig.savefig(fname)
        plt.close(fig)


    # Runs a single epoch training
    def train_epoch(self, t, trn_loader):
        print("Single epoch training")

    # Contains the evaluation code for evaluating the student
    def eval(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device))
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num
 
    # Returns the loss value
    def criterion(self, t, outputs, targets):
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

    def cecriterion(self, t, outputs, targets):
        return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets.type(torch.long))
    

    def lderloss(self, t, prev_aug_outputs, curr_aug_outputs, aug_targets):
        a1=1.0
        a2=1.0
        l2 = np.linalg.norm(torch.cat(prev_aug_outputs,dim=1).cpu().detach().numpy()-torch.cat(curr_aug_outputs, dim=1).cpu().detach().numpy())
        return (a1*l2)+(a2*torch.nn.functional.cross_entropy(torch.cat(curr_aug_outputs, dim=1), aug_targets.type(torch.long)))
        #return 10*torch.nn.functional.cross_entropy(torch.cat(curr_aug_outputs, dim=1), aug_targets.type(torch.long))
