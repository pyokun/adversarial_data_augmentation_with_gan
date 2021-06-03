import os
import sys
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import BaseNetwork as Base
from util import utils


from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR




sys.path.append("..")

class InterpolationGAN(nn.Module):

    def __init__(self, conf, device, is_train=True):
        super(InterpolationGAN, self).__init__()
        self.conf = conf
        self.device = device

        # Generator 
        self.G_ST = Base.Stoch_Generator(24, 3, 3, conf.g_conv_dim, 
                                        False, conf.n_res_blocks)                 
        self.G_TS = Base.Stoch_Generator(24, 3, 3, conf.g_conv_dim, 
                                        False, conf.n_res_blocks)                              


        # Domainess encoded latent vector Generator 


        if is_train:
            # Discriminator 
            self.D_S = Base.Discriminator(3, conf.d_conv_dim) 
            self.D_T1 = Base.Discriminator(3, conf.d_conv_dim) 
            self.D_T2 = Base.Discriminator(3, conf.d_conv_dim) 
            self.D_T3 = Base.Discriminator(3,conf.d_conv_dim)
            g_params=list(self.G_ST.parameters())+ list(self.G_TS.parameters())
            d_params=list(self.D_S.parameters())+ list(self.D_T1.parameters())+list(self.D_T2.parameters())+list(self.D_T3.parameters())

            # Criterion
            self.optimizer_G = torch.optim.Adam(g_params,
                                                lr=conf.lr, betas=(conf.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(d_params,
                                                lr=conf.lr, betas=(conf.beta1, 0.999))
            self.criterionGAN = nn.MSELoss(reduction='mean')
            self.criterion_cycle = nn.L1Loss(reduction='mean')

            # Answer for discriminator
            self.ans_real = Variable(torch.ones([self.conf.batch_size, 1, 14, 14]), requires_grad=False).to(device)
            self.ans_fake = Variable(torch.zeros([self.conf.batch_size, 1, 14, 14]), requires_grad=False).to(device)

    def set_input(self, input, epoch):
        ''' 
            Iteration
            size : batch, channel, height, widt
        '''
        self.real_S = Variable(input['S_img']).to(self.device)
        self.real_T1 = Variable(input['T1_img']).to(self.device)
        self.real_T2 = Variable(input['T2_img']).to(self.device)
        self.real_T3 = Variable(input['T3_img']).to(self.device)
        
        if epoch % 4==0:
            self.domainess = torch.tensor([[1.0,0.0,0.0]]).float()
        elif epoch % 4==1:
            self.domainess = torch.tensor([[0.0,1.0,0.0]]).float()
        elif epoch % 4==2:
            self.domainess = torch.tensor([[0.0,0.0,1.0]]).float()

        else :
            self.domainess=torch.rand(1,3).cuda()
            self.domainess=F.normalize(self.domainess,p=1,dim=1)
        self.z = self.domainess.view(1,3,1,1).repeat(1,8,1,1).cuda()











    def set_requires_grad(self, model_list, requires_grad=False):
        """
        """
        if not isinstance(model_list, list):
            model_list = [model_list]
        for model in model_list:
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = requires_grad

    def forward(self, x, d):
        self.G_ST.eval()
        Z = d.view(d.size(0),3,1,1).repeat(1,8,1,1)                       # domainess Z (1, 16, 1, 1)
        fake_T = self.G_ST(x,Z)                                    # 
        return fake_T


    def train(self):
        self.D_S.train()
        self.D_T1.train()
        self.D_T2.train()
        self.D_T3.train()
        self.G_ST.train()
        self.G_TS.train()


        # Make flow S to T
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()
        

        # Train G
        self.fake_T = self.G_ST(self.real_S, self.z)                                            # S
        self.recons_S = self.G_TS(self.fake_T, self.z)                                          # fake_T
        


        self.loss_cycle_S = self.criterion_cycle(self.recons_S, self.real_S)*self.conf.lambda_cycle
        self.loss_D_T1 = self.criterionGAN(self.D_T1(self.fake_T), self.ans_real)
        self.loss_D_T2 = self.criterionGAN(self.D_T2(self.fake_T), self.ans_real)
        self.loss_D_T3 = self.criterionGAN(self.D_T3(self.fake_T), self.ans_real)

        self.loss_G_S2T = (self.domainess[0][0])*self.loss_D_T1 + self.domainess[0][1]*self.loss_D_T2+ self.domainess[0][2]*self.loss_D_T3 + self.loss_cycle_S
        self.loss_G_S2T.backward()
        self.optimizer_G.step()

        # Train D
        loss_real_dt1 = self.criterionGAN(self.D_T1(self.real_T1), self.ans_real)                # Real S images
        loss_fake_dt1 = self.criterionGAN(self.D_T1(self.fake_T.detach()), self.ans_fake)       # S
        self.loss_DT1 = loss_real_dt1 + loss_fake_dt1
        
        loss_real_dt2 = self.criterionGAN(self.D_T2(self.real_T2), self.ans_real)                # Real T images
        loss_fake_dt2 = self.criterionGAN(self.D_T2(self.fake_T.detach()), self.ans_fake)       # S
        self.loss_DT2 = loss_real_dt2 + loss_fake_dt2

        loss_real_dt3 = self.criterionGAN(self.D_T3(self.real_T3), self.ans_real)                # Real T images
        loss_fake_dt3 = self.criterionGAN(self.D_T3(self.fake_T.detach()), self.ans_fake)       # S
        self.loss_DT3 = loss_real_dt3 + loss_fake_dt3


        
        self.loss_D_S2T = (self.domainess[0][0])*self.loss_DT1 + self.domainess[0][1]*self.loss_DT2+ self.domainess[0][2]*self.loss_DT3
        self.loss_D_S2T.backward()
        self.optimizer_D.step()

        torch.cuda.empty_cache()

        # Make flow T to S
        self.z1=1-self.z
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

        self.fake_S1 = self.G_TS(self.real_T1, self.z1)                                          # 
        self.fake_S2 = self.G_TS(self.real_T2,self.z1)
        self.fake_S3 = self.G_TS(self.real_T3,self.z1)
        self.recons_T1 = self.G_ST(self.fake_S1, self.z1)                                        # fa
        self.recons_T2 = self.G_ST(self.fake_S2, self.z1)
        self.recons_T3 = self.G_ST(self.fake_S3, self.z1)

        # Train G
        self.loss_cycle_T1 = self.criterion_cycle(self.recons_T1, self.real_T1)*self.conf.lambda_cycle
        self.loss_cycle_T2 = self.criterion_cycle(self.recons_T2, self.real_T2)*self.conf.lambda_cycle
        self.loss_cycle_T3 = self.criterion_cycle(self.recons_T3, self.real_T3)*self.conf.lambda_cycle


        self.loss_G_S1 = self.criterionGAN(self.D_S(self.fake_S1), self.ans_real)
        self.loss_G_S2 = self.criterionGAN(self.D_S(self.fake_S2), self.ans_real)
        self.loss_G_S3 = self.criterionGAN(self.D_S(self.fake_S3), self.ans_real)
        self.loss_G_T2S =(1- self.domainess[0][0])*self.loss_G_S1+(1-self.domainess[0][1])*self.loss_G_S2+(1-self.domainess[0][2])*self.loss_G_S3+(1-self.domainess[0][0])*self.loss_cycle_T1+(1-self.domainess[0][1])*self.loss_cycle_T2+(1-self.domainess[0][2])*self.loss_cycle_T3
        self.loss_G_T2S.backward()
        self.optimizer_G.step()

        # Train D
        loss_S_real_S = self.criterionGAN(self.D_S(self.real_S), self.ans_real)                # Real S images

        loss_S_fake_S1 = self.criterionGAN(self.D_S(self.fake_S1.detach()), self.ans_fake)       # 
        self.loss_D_S1 = loss_S_real_S + loss_S_fake_S1

        loss_S_fake_S2 = self.criterionGAN(self.D_S(self.fake_S2.detach()), self.ans_fake)       # T
        self.loss_D_S2 = loss_S_real_S + loss_S_fake_S2

        loss_S_fake_S3 = self.criterionGAN(self.D_S(self.fake_S3.detach()), self.ans_fake)       # T
        self.loss_D_S3 = loss_S_real_S + loss_S_fake_S3


        
        self.loss_D_T2S = (1-self.domainess[0][0])*self.loss_D_S1 + (1-self.domainess[0][1])*self.loss_D_S2+ (1-self.domainess[0][2])*self.loss_D_S3
        self.loss_D_T2S.backward()
        self.optimizer_D.step()

        torch.cuda.empty_cache()



    def denorm(self, x): 
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    

    def get_plot_list(self):
        lin=torch.linspace(0,1,5)        
        out=[]
        for i in lin:
            out.append(torch.cat([i.unsqueeze(0),1-i.unsqueeze(0)],dim=0))
        return out

    def save_output(self,s_img,i):
        x_list=[]
        x_list.append(s_img.cpu())

        z_list=self.get_plot_list()
        self.G_ST.eval()
        with torch.no_grad():
            for z in [torch.tensor([1,0,0]),torch.tensor([0,1,0]),torch.tensor([0,0,1])]:
                z=z.float().cuda()
                z=z.view(1,3,1,1).repeat(1,8,1,1)
                fake_T=self.G_ST(s_img.cuda(), z)
                x_list.append(fake_T.cpu())
        x_concat=torch.cat(x_list,dim=3)
        out_path=os.path.join(self.conf.checkpoint_dir,'{}-img.jpg'.format(i))
        save_image(self.denorm(x_concat.data.cpu()), out_path, nrow=1, padding=0)






    def save(self, ckp_name):
        path = os.path.join(self.conf.checkpoint_dir, ckp_name)
        checkpoint = { 'G_ST': self.G_ST.state_dict(),
                       'G_TS': self.G_TS.state_dict(),
                    }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.G_ST.load_state_dict(checkpoint['G_ST'])
        self.G_TS.load_state_dict(checkpoint['G_TS'])
