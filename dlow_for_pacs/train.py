import os
import sys
import yaml
import time

import torch
import argparse
from pacs_loader import get_loader
from util.Logger import Logger
from util.utils import*
from model.InterpolationGAN_v3 import InterpolationGAN

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)

    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--lambda_cycle',type=float,default=10.0)
    parser.add_argument('--n_res_blocks',type=int,default=5)
    

    # training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=20001)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    
    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--checkpoint_dir', type=str, default='./3targets_pac')

    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_step', type=int , default=1000)


    parser.add_argument('--s_idx', type=str , default='p')
    parser.add_argument('--t1_idx',type=str,default='a')
    parser.add_argument('--t2_idx',type=str,default='c')

    conf = parser.parse_args()



    # Reading configuration file

    # Checkpoint directory
    if not os.path.isdir(conf.checkpoint_dir):
        os.mkdir(conf.checkpoint_dir)
    
    device = torch.device('cuda')

    print("Using ", device)

    # Model 
    model = InterpolationGAN(conf, device=device)
    print_network(model)
    model = torch.nn.DataParallel(model, output_device=1)
    model.to(device)


    sys.stdout.write("Loading the data...")
    loaders=get_loader([conf.s_idx,conf.t1_idx,conf.t2_idx],256,256,2,'train',1)
    
    s_loader=iter(loaders[0])
    t_loader1=iter(loaders[1])
    t_loader2=iter(loaders[2])
    t_loader3=iter(loaders[3])
    iter_per_epoch = min(len(s_loader),len(t_loader1),len(t_loader2),len(t_loader3))

    fixed_t2=t_loader2.next()[0]
    fixed_t1 = t_loader1.next()[0]
    fixed_s = s_loader.next()[0]
    
    sys.stdout.write(" -done")
    sys.stdout.flush()

    #logger = Logger(train_conf)

    # Train
    for epoch in range(conf.train_iters):
        if (epoch+1) % (iter_per_epoch-1) == 0:
            s_loader=iter(loaders[0])
            t_loader1=iter(loaders[1])
            t_loader2=iter(loaders[2])
            t_loader3=iter(loaders[3])
        s_img,_=s_loader.next()            
        t1_img,_=t_loader1.next()
        t2_img,_=t_loader2.next()
        t3_img,_=t_loader3.next()


        # Actual training
        model.module.set_input({'S_img':s_img,'T1_img':t1_img,'T2_img':t2_img,'T3_img':t3_img},epoch)
        model.module.train()
        
        # Logging on terminal
        sys.stdout.write('\nEpoch %03d/%03d'%(epoch, conf.train_iters))
        sys.stdout.flush()

        # Logging on tensorboard
        '''
        if epoch % 5000 == 0:
            loss_log, img_log = model.module.get_data_for_logging()
            logger.log(loss_log, img_log)
        '''

        if (epoch % 5000) == 0: 
            model.module.save(str(epoch)+'.pth.tar')
            model.module.save_output(fixed_s,epoch)


   
