import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class InstanceNorm(nn.Module): 
    def __init__(self, num_features, affine=True, eps=1e-5):
        super(InstanceNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        self.scale = Parameter(torch.Tensor(num_features))
        self.shift = Parameter(torch.Tensor(num_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.scale.data.normal_(mean=0., std=0.02)
            self.shift.data.zero_()

    def forward(self, input):
        size = input.size()
        x_reshaped = input.view(size[0], size[1], size[2]*size[3])
        mean = x_reshaped.mean(2, keepdim=True)
        centered_x = x_reshaped - mean
        std = torch.rsqrt((centered_x ** 2).mean(2, keepdim=True) + self.eps)
        norm_features = (centered_x * std).view(*size)

        if self.affine:
            output = norm_features * self.scale[:,None,None] + self.shift[:,None,None]
        else:
            output = norm_features

        return output


class ResBlock(nn.Module):
    def __init__(self, input_nc, use_dropout):
        super(ResBlock, self).__init__()
        res_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, input_nc, 3),
            InstanceNorm(input_nc),
            nn.ReLU(inplace=True) 
        ]

        if use_dropout:
            res_block += [nn.Dropout(0.5)]

        res_block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, input_nc, 3),
            InstanceNorm(input_nc)
        ]
        self.res_block = nn.Sequential(*res_block)

    def forward(self, inputs):
        return self.res_block(inputs) + inputs


class TwoInputModule(nn.Module):
    def forward(self, input1, input2):
        raise NotImplementedError


class MergeModule(TwoInputModule):
    def __init__(self, module1, module2):

        super(MergeModule, self).__init__()
        self.module1 = module1
        self.module2 = module2

    def forward(self, input1, input2):
        output1 = self.module1.forward(input1)
        output2 = self.module2.forward(output1, input2)
        return output2


class TwoInputSequential(nn.Sequential, TwoInputModule):
    def __init__(self, *args):
        super(TwoInputSequential, self).__init__(*args)

    def forward(self, input1, input2):

        for module in self._modules.values():
            if isinstance(module, TwoInputModule):
                input1 = module.forward(input1, input2)
            else:
                input1 = module.forward(input1)
        return input1


class CondInstanceNorm(TwoInputModule):
    def __init__(self, x_dim, z_dim, eps=1e-5):

        super(CondInstanceNorm, self).__init__()
        self.eps = eps
        self.scale_conv = nn.Sequential(    nn.Conv2d(z_dim, x_dim, kernel_size=1, padding=0, bias=True),
                                            nn.ReLU(True)   )
        self.shift_conv = nn.Sequential(    nn.Conv2d(z_dim, x_dim, kernel_size=1, padding=0, bias=True),
                                            nn.ReLU(inplace=True)   ) 

    def forward(self, input, noise):

        shift = self.shift_conv.forward(noise)
        scale = self.scale_conv.forward(noise)
        
        size = input.size() 
        x_reshaped = input.view(size[0], size[1], size[2]*size[3])
        
        mean = x_reshaped.mean(2, keepdim=True)   
        var = x_reshaped.var(2, keepdim=True)      
        std =  torch.rsqrt(var + self.eps)  
        
        norm_features = ((x_reshaped - mean) * std).view(*size)
        output = norm_features * scale + shift
        return output


class CINResnetBlock(TwoInputModule):
    def __init__(self, x_dim, z_dim, norm_layer, use_dropout, use_bias):
        super(CINResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(x_dim, z_dim, norm_layer, use_dropout, use_bias)
        self.relu = nn.LeakyReLU(inplace=True)

        for idx, module in enumerate(self.conv_block):
            self.add_module(str(idx), module)

    def build_conv_block(self, x_dim, z_dim, norm_layer, use_dropout, use_bias):
        conv_block = [ nn.ReflectionPad2d(1) ]
        
        conv_block += [
            MergeModule(
                nn.Conv2d(x_dim, x_dim, kernel_size=3, padding=0, bias=use_bias),
                norm_layer(x_dim, z_dim)
            ),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReflectionPad2d(1)]
       
        conv_block += [nn.Conv2d(x_dim, x_dim, kernel_size=3, padding=0, bias=use_bias),
                       InstanceNorm(x_dim, affine=True)]

        return TwoInputSequential(*conv_block)

    def forward(self, x, noise):
        out = self.conv_block(x, noise)
        out = self.relu(x + out)
        return out
