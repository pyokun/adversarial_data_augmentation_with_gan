import os
import sys
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import BaseNetwork as Base
from util import utils


from torchvision.utils import save_image






sys.path.append("..")

class InterpolationGAN(nn.Module):

    def __init__(self, conf, device, is_train=True):
        super(InterpolationGAN, self).__init__()
        self.conf = conf
        self.device = device

        # Generator 
        self.G_ST = Base.Stoch_Generator(32,3, 3, conf.g_conv_dim, 
                                        True, conf.n_res_blocks)                 
        self.G_TS = Base.Stoch_Generator(32, 3, 3, conf.g_conv_dim, 
                                        True, conf.n_res_blocks)                               



        if is_train:
            # Discriminator 
            self.D_S = Base.Discriminator(3,conf.d_conv_dim) 
            self.D_T1 = Base.Discriminator(3, conf.d_conv_dim) 
            self.D_T2 = Base.Discriminator(3, conf.d_conv_dim) 

            g_params= list(self.G_ST.parameters())+ list(self.G_TS.parameters())
            d_params=list(self.D_S.parameters())+ list(self.D_T1.parameters())+list(self.D_T2.parameters())

            # Criterion
            self.optimizer_G = torch.optim.Adam(g_params,
                                                lr=conf.lr, betas=(conf.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(d_params,
                                                lr=conf.lr, betas=(conf.beta1, 0.999))
            self.criterionGAN = nn.MSELoss(reduction='mean')
            self.criterion_cycle = nn.L1Loss(reduction='mean')

            # Answer for discriminator
            self.ans_real = Variable(torch.zeros([self.conf.batch_size, 1, 14, 14]), requires_grad=False).to(device)
            self.ans_fake = Variable(torch.ones([self.conf.batch_size, 1, 14, 14]), requires_grad=False).to(device)

    def set_input(self, input,epoch):
        ''' 
            Iteration
            size : batch, channel, height, widt
        '''
        self.real_S = Variable(input['S_img']).to(self.device)
        self.real_T1 = Variable(input['T1_img']).to(self.device)
        self.real_T2 = Variable(input['T2_img']).to(self.device)
        self.domainess = utils.get_domainess(epoch, self.conf.train_iters, 1).to(self.device)
        self.Z = self.domainess.view(1,2,1,1).repeat(1,16,1,1)

    def set_requires_grad(self, model_list, requires_grad=False):
        """
        """
        if not isinstance(model_list, list):
            model_list = [model_list]
        for model in model_list:
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = requires_grad

    def forward(self, input, domainess):
        d = torch.tensor([[domainess]])                                 
        Z = d.view(1,2,1,1).repeat(1,16,1,1)                       # domainess Z (1, 16, 1, 1)
        real_S = Variable(input).to(self.device)
        fake_T = self.G_ST(real_S,Z)                                    # 
        return fake_T

    def train(self):
        self.D_S.train()
        self.D_T1.train()
        self.D_T2.train()
        self.G_ST.train()
        self.G_TS.train()


        # Make flow S to T
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()
        


        # Train G
        self.fake_T = self.G_ST(self.real_S, self.Z)                                            # S
        self.recons_S = self.G_TS(self.fake_T, self.Z)                                          # fake_T
        


        self.loss_cycle_S = self.criterion_cycle(self.recons_S, self.real_S)*self.conf.lambda_cycle
        self.loss_D_T1 = self.criterionGAN(self.D_T1(self.fake_T), self.ans_real)
        self.loss_D_T2 = self.criterionGAN(self.D_T2(self.fake_T), self.ans_real)
        self.loss_G_S2T = (self.domainess[0][0])*self.loss_D_T1 + self.domainess[0][1]*self.loss_D_T2 + self.loss_cycle_S
        self.loss_G_S2T.backward()
        self.optimizer_G.step()

        # Train D
        loss_real_dt1 = self.criterionGAN(self.D_T1(self.real_T1), self.ans_real)                # Real S images
        loss_fake_dt1 = self.criterionGAN(self.D_T1(self.fake_T.detach()), self.ans_fake)       # S
        self.loss_DT1 = loss_real_dt1 + loss_fake_dt1
        
        loss_real_dt2 = self.criterionGAN(self.D_T2(self.real_T2), self.ans_real)                # Real T images
        loss_fake_dt2 = self.criterionGAN(self.D_T2(self.fake_T.detach()), self.ans_fake)       # S
        self.loss_DT2 = loss_real_dt2 + loss_fake_dt2
        
        self.loss_D_S2T = (self.domainess[0][0])*self.loss_DT1 + self.domainess[0][1]*self.loss_DT2
        self.loss_D_S2T.backward()
        self.optimizer_D.step()

        torch.cuda.empty_cache()

        # Make flow T to S
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

        self.Z_1 =  1-self.Z         # domainess 1 - Z (1, 16, 1, 1)
        self.fake_S1 = self.G_TS(self.real_T1, self.Z_1)                                          # 
        self.fake_S2 = self.G_TS(self.real_T2,self.Z_1)
        self.recons_T1 = self.G_ST(self.fake_S1, self.Z_1)                                        # fa
        self.recons_T2 = self.G_ST(self.fake_S2, self.Z_1) 
        # Train G
        self.loss_cycle_T1 = self.criterion_cycle(self.recons_T1, self.real_T1)*self.conf.lambda_cycle
        self.loss_cycle_T2 = self.criterion_cycle(self.recons_T2, self.real_T2)*self.conf.lambda_cycle
        self.loss_G_S1 = self.criterionGAN(self.D_S(self.fake_S1), self.ans_real)
        self.loss_G_S2 = self.criterionGAN(self.D_S(self.fake_S2), self.ans_real)
        self.loss_G_T2S = (1-self.domainess[0][0])*self.loss_G_S1+(1-self.domainess[0][1])*self.loss_G_S2 +(1-self.domainess[0][0])*self.loss_cycle_T1+(1-self.domainess[0][1])*self.loss_cycle_T2
        self.loss_G_T2S.backward()
        self.optimizer_G.step()

        # Train D
        loss_S_real_S = self.criterionGAN(self.D_S(self.real_S), self.ans_real)                # Real S images

        loss_S_fake_S1 = self.criterionGAN(self.D_S(self.fake_S1.detach()), self.ans_fake)       # 
        self.loss_D_S1 = loss_S_real_S + loss_S_fake_S1

        #loss_S_real_S = self.criterionGAN(self.D_S(self.real_T), self.ans_real)                # Real T images
        loss_S_fake_S2 = self.criterionGAN(self.D_S(self.fake_S2.detach()), self.ans_fake)       # T
        self.loss_D_S2 = loss_S_real_S + loss_S_fake_S2
        
        self.loss_D_T2S = (1-self.domainess[0][0])*self.loss_D_S1 + (1-self.domainess[0][1])*self.loss_D_S2
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

        for z in [0.0,0.5,1.0]:
            z=z
            z=torch.tensor([[z,1-z]]).cuda()
            z=z.view(1,2,1,1).repeat(1,16,1,1)
            fake_T=self.G_ST(s_img.cuda(), z)
            x_list.append(fake_T.cpu())
        x_concat=torch.cat(x_list,dim=3)
        out_path=os.path.join(self.conf.checkpoint_dir,'{}-img.jpg'.format(i))
        save_image(self.denorm(x_concat.data.cpu()), out_path, nrow=1, padding=0)






    def save(self, ckp_name):
        path = os.path.join(self.conf.checkpoint_dir, ckp_name)
        checkpoint = { 'G_ST': self.G_ST.state_dict(),
                       'G_TS': self.G_TS.state_dict()
                    }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.G_S.load_state_dict(checkpoint['G_S'])
        self.G_T.load_state_dict(checkpoint['G_T'])