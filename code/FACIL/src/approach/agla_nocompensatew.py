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

import math

class Appr(Inc_Learning_Appr):

    def __init__(self, model, device, nepochs=160, lr=0.1, lr_min=1e-4, lr_factor=10, lr_patience=8, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        print("AGLA Incremental Learning")
        self.augmentor = None
        self.assessor = None
        
        self.transformed_data=None
        self.batch_size =  None
        
        self.prev_val_loader =  None
        self.mem_x =  None
        self.mem_y =  None
        
        self.trans_mem_x =  None
        self.trans_mem_y =  None
        self.prev_mem_outputs = None
        
        self.outer_epochs = 1
        self.inner_epochs = 1

        self.model_old = None

        self.bias_epochs =  100
        self.M = 5

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
            self.assessor = LSTMAssessor(3,self.device).to(self.device)


    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        #combine dataset in t>0
       
        if(t>0):
            self.model.eval()
            self.model.to(self.device)
            self.prev_mem_outputs = self.model(self.mem_x.view(self.mem_x.shape[0]*self.mem_x.shape[1],self.mem_x.shape[2],self.mem_x.shape[3],self.mem_x.shape[4]))
            
            self.generate_transformed_memory(t)
            self.trans_mem_x = self.trans_mem_x.to(self.device)
            self.trans_mem_y = self.trans_mem_y.to(self.device)
            self.prev_trans_mem_outputs = self.model(self.trans_mem_x)
           

        trn_comb_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        self.baseOptimizer = self._get_optimizer(self.model, self.lr)
        self.assessorOptimizer = self._get_optimizer(self.assessor, self.lr)


        self.generate_transformed_data(t, trn_comb_loader)
        trf_loader = torch.utils.data.DataLoader(self.transformed_data,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)


        best_acc = 0.0
        best_model = None

        if t > 0:
            tr_mem_x = self.mem_x.view(self.mem_x.shape[0]*self.mem_x.shape[1],self.mem_x.shape[2],self.mem_x.shape[3],self.mem_x.shape[4]).to(self.device)
            tr_mem_y = self.mem_y.view(self.mem_y.shape[0]*self.mem_y.shape[1]).to(self.device)

        #Train inner outer
        for e in range(self.nepochs):
            # if t > 0:
            #     self.compute_memory_aug_weight(t)
            for oe in range(self.outer_epochs):

                print("[Inner Loop] Train Assessor")
                loss = loss1 = loss2 = dloss = 0
                for ie in range(self.inner_epochs):

                    self.assessor.train()
                    # if t > 0:
                        

                    for images, targets in trf_loader:

                        outputs = self.model(images.to(self.device))                
                        if(t>0):
                            trans_mem_outputs_old = self.model_old(self.trans_mem_x.to(self.device))
                            curr_trans_mem_outputs = self.model(self.trans_mem_x.to(self.device)) 
                            loss1 = self.cecriterion(t, outputs, targets.to(self.device).type(torch.long)) + self.cecriterion(t, curr_trans_mem_outputs, self.trans_mem_y)
                            loss2 = self.dercriterion(t, self.prev_trans_mem_outputs, curr_trans_mem_outputs, self.trans_mem_y) 
                            dloss = self.distcriterion(t, curr_trans_mem_outputs, self.trans_mem_y.to(self.device), trans_mem_outputs_old)
                            loss = loss1+loss2+dloss
                        else:
                            loss = self.cecriterion(t, outputs, targets.to(self.device).type(torch.long))

                        self.assessorOptimizer.zero_grad()
                        loss.backward()
                        self.assessorOptimizer.step()
                    print('I-Epoch :{:3d} {:3d} {:3d} loss:{:5.1f} {:5.1f} {:5.1f} {:5.1f}'.format(e+1,oe+1, oe+1,loss, loss1, loss2, dloss))            

                
                print("[Outer Loop] Train Base ")
                self.model.train()
                #self.assessor.eval()
                clock0 = time.time()
                for images, targets in trn_comb_loader:
                    #print(images[0])
                    #print(targets)
                    c_lr = self.assessor(images.to(self.device))
                    self.model.to(self.device)
                    outputs = self.model(images.to(self.device))

                    if(t>0):
                        mem_outputs_old = self.model_old(tr_mem_x)
                        curr_mem_outputs = self.model(tr_mem_x)
                        loss1 = self.cecriterion(t, outputs, targets.to(self.device)) + self.cecriterion(t, curr_mem_outputs, tr_mem_y)
                        loss2 = self.dercriterion(t, self.prev_mem_outputs, curr_mem_outputs, tr_mem_y) 
                        dloss = self.distcriterion(t, curr_mem_outputs, tr_mem_y, mem_outputs_old)
                        loss = (c_lr[0] * loss1)+(c_lr[1] * loss2 * t) + (c_lr[2] * dloss * t)
                    else:
                        loss = c_lr[0] * self.cecriterion(t, outputs, targets.to(self.device))

                    self.baseOptimizer.zero_grad()
                    loss.backward()
                    self.baseOptimizer.step()
                    
                #Cek accuracy per epoch
                clock1 = time.time()
                val_loss, val_acc, _ = self.eval(t, val_loader)
                clock2 = time.time()
                print('| O-Epoch {:3d} {:3d}, time={:5.1f}s/{:5.1f}s | c_lr={:.3f} {:.3f} {:.3f}|Valid: loss={:.3f} {:.3f} {:.3f} {:.3f} {:.3f}, TAw acc={:5.1f}% |'.format(
                e+1, oe+1, clock1 - clock0, clock2 - clock1, c_lr[0], c_lr[1],c_lr[2], val_loss, loss, loss1, loss2, dloss, 100 * val_acc), end='\n')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=val_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * val_acc, group="train")  

                
                if val_acc > best_acc:                
                # best_loss = val_loss
                    best_acc = val_acc
                    best_model = self.model.get_copy()
        
        self.model.set_state_dict(best_model)

        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

        if (t > 0):
            self.exemplars_dataset.collect_exemplars(self.model, trn_comb_loader, val_loader.dataset.transform) 
            self.preprocess_memory(t, trn_comb_loader, val_loader.dataset.transform)
        else:
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)
            self.preprocess_memory(t, trn_loader, val_loader.dataset.transform)


    def preprocess_memory(self, t, trn_loader, transform):
        if(t==0):
            self.exemplar_ratio = len(trn_loader.dataset.images) / len(self.exemplars_dataset.images)

        isFirst=[True,True,True,True,True]
        
        tx = np.array([])
        ty = np.array([])
        
        tmp_x = [tx,tx,tx,tx,tx]
        tmp_y = [ty,ty,ty,ty,ty]

        with override_dataset_transform(self.exemplars_dataset, transform) as _ds:
            mem_loader = torch.utils.data.DataLoader(_ds, batch_size=trn_loader.batch_size, shuffle=False,
                                                        num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
            nt = 0
            for images, targets in mem_loader:
                images = images.detach().cpu()
                targets = targets.detach().cpu()

                if(isFirst[0]):
                    isFirst[0]=False
                    tmp_x[0] = images
                    tmp_y[0] = targets
                else:
                    tmp_x[0] = np.concatenate((tmp_x[0], images),axis=0)
                    tmp_y[0]= np.concatenate((tmp_y[0], targets),axis=0)

                if(++nt < self.exemplar_ratio):
                    x = images + (0.1**0.5)*torch.randn(images.shape[0],images.shape[1],images.shape[2],images.shape[3])
                    if(isFirst[1]):
                        isFirst[1]=False
                        tmp_x[1] = x
                        tmp_y[1] = targets
                    else:
                        tmp_x[1] = np.concatenate((tmp_x[1], x),axis=0)
                        tmp_y[1] = np.concatenate((tmp_y[1], targets),axis=0)

                if(++nt < self.exemplar_ratio):
                    x = images + torch.rand(images.shape[0],images.shape[1],images.shape[2],images.shape[3])
                    if(isFirst[2]):
                        isFirst[2]=False
                        tmp_x[2] = x
                        tmp_y[2] = targets
                    else:
                        tmp_x[2] = np.concatenate((tmp_x[2], x),axis=0)
                        tmp_y[2] = np.concatenate((tmp_y[2], targets),axis=0)

                if(++nt < self.exemplar_ratio):
                    hflipper = torchvision.transforms.RandomHorizontalFlip(p=1.0)
                    x = hflipper(images)
                    if(isFirst[3]):
                        isFirst[3]=False
                        tmp_x[3] = x
                        tmp_y[3] = targets
                    else:
                        tmp_x[3] = np.concatenate((tmp_x[3], x),axis=0)
                        tmp_y[3] = np.concatenate((tmp_y[3], targets),axis=0)


                if(++nt < self.exemplar_ratio):
                    vflipper = torchvision.transforms.RandomVerticalFlip(p=1.0)
                    x = vflipper(images)
                    if(isFirst[4]):
                        isFirst[4]=False
                        tmp_x[4] = x
                        tmp_y[4] = targets
                    else:
                        tmp_x[4] = np.concatenate((tmp_x[4], x),axis=0)
                        tmp_y[4] = np.concatenate((tmp_y[4], targets),axis=0)

        self.mem_x = torch.tensor(np.array(tmp_x)).to(self.device)
        self.mem_y = torch.tensor(np.array(tmp_y)).to(self.device)

        print("Memory shape:")
        print(self.mem_x.shape)
        print(self.mem_y.shape)

    # def compute_memory_aug_weight(self, t):
        
    #     temperature=0.5
    #     # t = torch.Tensor([])
    #     feat0 = self.model(self.mem_x[0], return_features=True)[1]
    #     N = feat0.shape[0]
    #     mem_output = np.zeros((self.M,N,feat0.shape[1]))
    #     # mem_output[0,:,:] = feat0
    #     # print("mem output size: ")
    #     # print(mem_output.shape)

    #     for i in range(0, self.M):
    #         mem_output[i,:,:] = self.model(self.mem_x[i], return_features=True)[1].cpu().detach().numpy()

    #     mu_mem_output =  np.mean(mem_output,axis=0)

    #     for i in range(0, self.M):
    #         mem_output[i,:,:] = mem_output[i,:,:] - mu_mem_output 

    #     sigma=0.0
    #     zminu = np.zeros((self.M,N))
    #     for i in range(0, self.M):
    #         for j in range(0, N):
    #             zminu[i,j] = np.multiply(mem_output[i,j,:],mem_output[i,j,:]).sum()
    #             sigma += zminu[i,j]

    #     sigma = sigma / (N * self.M)


    #     w = np.zeros((self.M,N))
    #     sum_w = 0.0
    #     # for i in range(0, self.M):
    #     #     for j in range(0, N):
    #     #         w[i,j] = math.exp(-1 * 1/(sigma*temperature) * zminu[i,j])
    #     #         sum_w += w[i,j]
    #     w = np.exp(-1 / (sigma*temperature) * zminu)
    #     sum_w = w.sum(axis=1).sum(axis=0)

    #     w = w / sum_w
    #     # self.mem_w = torch.tensor(w.sum(axis=1).sum(axis=0)).to(self.device)
    #     self.mem_w = w.sum(axis=1).sum(axis=0)
    #     # print("cek w constant:")
    #     # print(self.mem_w)
    
    def compute_memory_aug_weight(self, t):
        
        temperature=0.5
        # t = torch.Tensor([])
        feat0 = self.model(self.mem_x[0], return_features=True)[1].cpu()
        N = feat0.shape[0]
        mem_output = torch.zeros(self.M,N,feat0.shape[1])
        mem_output[0,:,:] = feat0
        # print("mem output size: ")
        # print(mem_output.shape)

        for i in range(1, self.M):
            mem_output[i,:,:] = self.model(self.mem_x[i], return_features=True)[1].cpu()

        mu_mem_output =  torch.mean(mem_output,axis=0)

        for i in range(0, self.M):
            mem_output[i,:,:] = mem_output[i,:,:] - mu_mem_output 

        sigma=0.0
        zminu = torch.zeros(self.M,N)
        for i in range(0, self.M):
            for j in range(0, N):
                zminu[i,j] = (mem_output[i,j,:]*mem_output[i,j,:]).sum()
                sigma += zminu[i,j]
        sigma = sigma / (N * self.M)


        w = torch.zeros(self.M,N)
        sum_w = 0.0
        for i in range(0, self.M):
            for j in range(0, N):
                w[i,j] = math.exp(-1 * 1/(sigma*temperature) * zminu[i,j])
                # sum_w += w[i,j]

        # w = torch.mul(w, (1/sum_w))
        w = w / w.view(self.M*N).sum()

        self.mem_w = w.view(self.M*N).sum()


    def post_train_process(self, t, trn_loader):
        print("Post Train Process")

    def generate_transformed_data(self, t, trn_loader):
        notSaved=True
        isEmpty=True
        trf_imgs_x = np.array([])
        trf_imgs_y = np.array([])
      
        for images, targets in trn_loader:
            x1 = images.detach().cpu()
            x1 = x1 + (0.1**0.5)*torch.randn(x1.shape[0],x1.shape[1],x1.shape[2],x1.shape[3])
            #Random RGB
            x1 = x1 + torch.rand(x1.shape[0],x1.shape[1],x1.shape[2],x1.shape[3])
            #Invery
            x1 = torchvision.transforms.functional.invert(x1)

            if(notSaved):
                print("Images size")
                print(images.cpu().shape)
                self.save_mem_image(images.cpu(),"trf_ori.jpg")
                self.save_mem_image(x1,"3xtrf_invert.jpg")
                notSaved=False

            if(isEmpty):
                trf_imgs_x = x1
                trf_imgs_y = targets.cpu()
                isEmpty=False
            else:
                trf_imgs_x = np.concatenate((trf_imgs_x, x1),axis=0)
                trf_imgs_y = np.concatenate((trf_imgs_y, targets.cpu()),axis=0)
                

        print("Random Transformation Finish!!")
        print(trf_imgs_x.shape)
        print(trf_imgs_y.shape)
        tensor_x = torch.Tensor(trf_imgs_x) # transform to torch tensor
        tensor_y = torch.Tensor(trf_imgs_y)

        
        self.transformed_data = TensorDataset(tensor_x,tensor_y) # create your datset




    def generate_transformed_memory(self, t):
        torch.cuda.empty_cache()

        isFirst=True
        x = self.mem_x.view(self.mem_x.shape[0]*self.mem_x.shape[1],self.mem_x.shape[2],self.mem_x.shape[3],self.mem_x.shape[4]).detach().cpu()
        y = self.mem_y.view(self.mem_y.shape[0]*self.mem_y.shape[1]).detach().cpu()

        #Random Gauss
        x = x + (0.1**0.5)*torch.randn(x.shape[0],x.shape[1],x.shape[2],x.shape[3])
        #Random RGB
        x = x + torch.rand(x.shape[0],x.shape[1],x.shape[2],x.shape[3])
        x = torchvision.transforms.functional.invert(x)

        self.trans_mem_x = x;
        self.trans_mem_y = y;
            

    def save_mem_image(self, x, fname):
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
    
    def distcriterion(self, t, outputs, targets, outputs_old=None):
        g = torch.sigmoid(torch.cat(outputs[:t], dim=1))
        q_i = torch.sigmoid(torch.cat(outputs_old[:t], dim=1))
        loss = sum(torch.nn.functional.binary_cross_entropy(g[:, y], q_i[:, y]) for y in
                                    range(sum(self.model.task_cls[:t])))
        return loss

    def dercriterion(self, t, prev_mem_outputs, curr_mem_outputs, mem_targets):
        a1=0.1
        a2=0.1
        l2 = np.linalg.norm(torch.cat(prev_mem_outputs,dim=1).cpu().detach().numpy()-torch.cat(curr_mem_outputs, dim=1).cpu().detach().numpy())
        return (a1*l2)+(a2*torch.nn.functional.cross_entropy(torch.cat(curr_mem_outputs, dim=1), mem_targets.type(torch.long)))
        #return 10*torch.nn.functional.cross_entropy(torch.cat(curr_mem_outputs, dim=1), mem_targets.type(torch.long))


class LSTMAssessor(nn.Module):

    def __init__(self,numChannels,device):

        super(LSTMAssessor, self).__init__()
        self.device = device
        
        # self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=numChannels,kernel_size=(3, 3))
        # self.relu1 = nn.ReLU()
        # self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # self.conv2 = nn.Conv2d(in_channels=numChannels, out_channels=64,kernel_size=(3, 3))
        # self.relu2 = nn.ReLU()
        # self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        res18 = torchvision.models.resnet18(pretrained=False)
        modules = list(res18.children())[:-1]
        self.feature_extractor=nn.Sequential(*modules)
        
        self.lstm= nn.LSTM(input_size=512,hidden_size=512,num_layers=2,batch_first=True)
        self.fc1 = nn.Linear(in_features=512,out_features=512)
        self.fc2 = nn.Linear(in_features=512,out_features=3)
        self.init_param()

        #print(self)

    def init_param(self):
        for name, param in self.named_parameters(): 
            torch.nn.init.normal_(param); 
    
    def forward(self, x):
        #x = torch.flatten(x)
        # x = self.conv1(x)
        # x = self.relu1(x)
        # x = self.maxpool1(x)

        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.maxpool2(x)
        x = self.feature_extractor(x)
        # print("The shape of x: ", end='')
        # print(x.shape)

        # x = x.view(x.shape[0],x.shape[1]*x.shape[2], x.shape[3])
        x = x.view(x.shape[0],x.shape[3],x.shape[1]*x.shape[2])

        h0 = torch.zeros(2, x.size(0), 512).to(self.device)
        c0 = torch.zeros(2, x.size(0), 512).to(self.device)

        out,_ = self.lstm(x,(h0,c0))
        out = out[:,-1,:]
        out = self.fc1(out)
        out = self.fc2(out)
        output = torch.sigmoid(out)
        output = np.mean(output.cpu().detach().numpy(),axis=0)
        return output

