import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import utils
from .Modules import*

sys.path.append("..")

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, use_dropout=False, n_res_blocks=6):
        super(Generator, self).__init__()
        # Initial convolution block       
        model = [    nn.ReflectionPad2d(3), # Reference from CycleGAN
                     nn.Conv2d(input_nc, ngf, 7),
                     InstanceNorm(ngf),
                     nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = ngf
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        InstanceNorm(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_res_blocks):
            model += [ ResBlock(in_features, use_dropout) ]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        InstanceNorm(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)
        utils.init_weights(self)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf):
        super(Discriminator, self).__init__()

        norm_layer = InstanceNorm

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1),
                    norm_layer(ndf*2), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=1),
                    norm_layer(ndf*4), 
                    nn.LeakyReLU(0.2, inplace=True) ]
  

        # Return 1 channel prediction map
        model += [  nn.Conv2d(ndf*4, 1, 4, padding=1)   ]

        self.model = nn.Sequential(*model)
        utils.init_weights(self)

    def forward(self, x): 
        return self.model(x)

class Stoch_Generator(nn.Module):
    def __init__(self, nlatent, input_nc, output_nc, ngf=64, use_dropout=False, n_blocks=9):
        super(Stoch_Generator, self).__init__()
        norm_layer = CondInstanceNorm

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, stride=1, bias=True),
            norm_layer(ngf, nlatent),
            nn.ReLU(True),

            nn.Conv2d(ngf, 2*ngf, kernel_size=3, padding=1, stride=1, bias=True),
            norm_layer(2*ngf, nlatent),
            nn.ReLU(True),

            nn.Conv2d(2*ngf, 4*ngf, kernel_size=3, padding=1, stride=2, bias=True),
            norm_layer(4*ngf, nlatent),
            nn.ReLU(True)
        ]
        
        for i in range(n_blocks):
            model += [  CINResnetBlock(x_dim=4*ngf, z_dim=nlatent, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=True)   ]

        model += [
            nn.ConvTranspose2d(4*ngf, 2*ngf, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=True),
            norm_layer(2*ngf , nlatent),
            nn.ReLU(True),

            nn.Conv2d(2*ngf, ngf, kernel_size=3, padding=1, stride=1, bias=True),
            norm_layer(ngf, nlatent),
            nn.ReLU(True),

            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3),
            nn.Tanh()
        ]

        self.model = TwoInputSequential(*model)
        utils.init_weights(self)

    def forward(self, input, noise):
        return self.model(input, noise)


class Latent_Encoder(nn.Module):
    def __init__(self, nlatent, input_nc, nef, norm_layer):
        super(Latent_Encoder, self).__init__()
        
        use_bias = False
        kw = 3

        sequence = [
            nn.Conv2d(input_nc, nef, kernel_size=kw, stride=2, padding=1, bias=True),
            nn.ReLU(True),

            nn.Conv2d(nef, 2*nef, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(2*nef),
            nn.ReLU(True),

            nn.Conv2d(2*nef, 4*nef, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(4*nef),
            nn.ReLU(True),

            nn.Conv2d(4*nef, 8*nef, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(8*nef),
            nn.ReLU(True),

            nn.Conv2d(8*nef, 8*nef, kernel_size=4, stride=1, padding=0, bias=use_bias),
            norm_layer(8*nef),
            nn.ReLU(True),

        ]

        self.conv_modules = nn.Sequential(*sequence)
        self.enc_mu = nn.Conv2d(8*nef, nlatent, kernel_size=1, stride=1, padding=0, bias=True)
        self.enc_logvar = nn.Conv2d(8*nef, nlatent, kernel_size=1, stride=1, padding=0, bias=True)
        utils.init_weights(self)

    def forward(self, input):
        conv_out = self.conv_modules(input)
        mu = self.enc_mu(conv_out)
        logvar = self.enc_logvar(conv_out)
        return (mu.view(mu.size(0), -1), logvar.view(logvar.size(0), -1))


class Latent_Discriminator(nn.Module):
    def __init__(self, nlatent, ndf):
        super(Latent_Discriminator, self).__init__()

        self.nlatent = nlatent

        use_bias = True
        sequence = [
            nn.Linear(nlatent, ndf),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, True),

            nn.Linear(ndf, ndf),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, True),

            nn.Linear(ndf, ndf),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, True),

            nn.Linear(ndf, 1)
        ]

        self.model = nn.Sequential(*sequence)
        utils.init_weights(self)

    def forward(self, input):
        if input.dim() == 4:
            input = input.view(input.size(0), self.nlatent)
        return self.model(input)