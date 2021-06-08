# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
sys.path.append('./../../')
import time
import uuid
import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.utils.data
from torch.utils.data import DataLoader,Dataset
from mnist_m import MNISTM
from digits_classification import hparams_registry
from digits_classification import algorithms
from digits_classification.lib import misc
from digits_classification.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domain_style_transfer import domain_transfer,minibatch_transfer
from syn_digits import Loadsyn
from torchvision.utils import save_image,make_grid
import torch.nn.functional as F
from dlow_model.InterpolationGAN import InterpolationGAN




class code_generate(nn.Module):
    def __init__(self,in_dim):
        super(code_generate,self).__init__()
        self.fc=nn.Linear(in_dim,in_dim)
        self.m=nn.Softmax(dim=1)
    def forward(self,z):
        x= self.fc(z)
        x=self.m(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="digits")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=11,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=6001,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=50,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="result/result_0")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0)
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')

    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)

    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--lambda_cycle',type=float,default=10.0)
    parser.add_argument('--n_res_blocks',type=int,default=3)
    parser.add_argument('--dlow_name', type=str, default="generator/dlow0")
    parser.add_argument('--source_idx', type=int, default=1) #we use id to denote four digits dataset, 1 for MNIST, 2 for SVHN, 3 for MNISTM, 4 for syndigits
                                                              # source_idx： source domain in DLOW
    parser.add_argument('--additional_idx1', type=int, default=2) # remaining training datasets's idx
    parser.add_argument('--additional_idx2', type=int, default=3) # remaining training datasets's idx
    parser.add_argument('--test_idx', type=int, default=4) #unseen test dataset's idx


    args = parser.parse_args()
    device = torch.device('cuda')
    #loade generator
    dlow_path=args.dlow_name
    gen = InterpolationGAN(args, device=device,is_train=False)
    gen.G_ST.to(device)
    gen.load(dlow_path)



    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
        print('cuda')
    else:
        device = "cpu"


    # prepare digits data

    trans = transforms.Compose([transforms.Resize(size=[32, 32]),transforms.ToTensor() ,transforms.Lambda(lambda x: x.repeat(3, 1, 1)),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])
    trans2=transforms.Compose([transforms.Resize(size=[32, 32]),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])


    current_iter=0

    uda_splits = []

    syn_dig=Loadsyn()
    mnist=torchvision.datasets.MNIST("./",train=True,transform=trans,download=True)
    mnistm = MNISTM("./",train=True,transform=trans2,download=True)
    svhn = torchvision.datasets.SVHN("./",transform=trans2,download=True)

    dataset_list={1:mnist,2:svhn,3:mnistm,4:syn_dig}
    uda_splits.append((dataset_list[args.test_idx],None))
        
    in_splits = []
    out_splits = []
    add_in_splits=[]
    train_datasets = [dataset_list[args.source_idx]]
    additional_datasets=[dataset_list[args.additional_idx1],dataset_list[args.additional_idx2]]

    for env_i,env in enumerate(train_datasets):
        out,in_=misc.split_dataset(env,int(len(env)*args.holdout_fraction),misc.seed_hash(args.trial_seed, env_i))
        in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=64,
        num_workers=0)
        for i, (env, env_weights) in enumerate(in_splits)]

    additional_loders=[InfiniteDataLoader(
        dataset=add_dataset,
        weights=None,
        batch_size=64,
        num_workers=0)
        for i, add_dataset in enumerate(additional_datasets)]



    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=1000,
        num_workers=0)
        for env, _ in (in_splits  + uda_splits)]


    eval_weights = [None for _, weights in (in_splits+out_splits + uda_splits)]

    eval_loader_names = ['env{}_train'.format(i)
        for i in range(len(in_splits))]

    eval_loader_names = ['env{}_validation'.format(i)
        for i in range(len(out_splits))]


    eval_loader_names += ['env{}_test'.format(i)
        for i in range(len(uda_splits))]


    #define main model 
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class((3,32,32), 10,
        len(train_datasets), hparams)
    algorithm.to(device)
    #defind code generator
    code_gen=code_generate(3)


    code_gen.to(device)
    code_optimizer=torch.optim.SGD(code_gen.parameters(),lr=0.02)
    


    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(temp_dataset)/hparams['batch_size'] for temp_dataset in train_datasets])

    n_steps = args.steps or 6001
    checkpoint_freq = args.checkpoint_freq or 50

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": (3,32,32),
            "model_num_classes": 10,
            "model_num_domains": 1,
            "model_hparams": hparams,
            "model_dict": algorithm.cpu().state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    last_results_keys = None
    start_step = 0
    code_update=0

    #we report test acc with maximal validataion acc
    max_validataion_acc=0              
    corresponding_test_acc=0  
    report_test=False
    
    for step in range(start_step, n_steps):
        algorithm.to(device)
        step_start_time = time.time()


        if step==4000:
            for g in algorithm.optimizer.param_groups:
                g['lr']=g['lr']*0.1


        x,y = next(iter(train_loaders[0]))
        x=x.cuda()
        y=y.cuda()
        z=torch.randn(x.size(0),3).cuda()
        z=code_gen(z)

        current_code=z.view(x.size(0),-1)
        new_x=minibatch_transfer(gen,x,current_code)

        train_x=torch.cat([x,new_x],dim=0)
        train_y=torch.cat([y,y],dim=0)

        for i in range(len(additional_loders)):
            add_x,add_y=next(iter(additional_loders[i]))
            add_x=add_x.cuda()
            add_y=add_y.cuda()

            train_x=torch.cat([train_x,add_x],dim=0)
            train_y=torch.cat([train_y,add_y],dim=0)

        step_vals = algorithm.update([(train_x,train_y)])
        code_file_name='code_'+str(code_update)




        
        if step != 0 and step % 5 ==0:
            code_optimizer.zero_grad()
            loss=-F.cross_entropy(algorithm.predict(new_x),y)
            loss.backward()
            code_optimizer.step()

        if step != 0 and step % 500 ==0:
            code_update+=1
        

        if step % 500 ==0:
            fp='saved_image'+str(step)+'.jpg'
            fp=os.path.join(args.output_dir, fp)
            image_batch=make_grid((new_x+1.0)/2.0)
            save_image(image_batch,fp)



        with open(os.path.join(args.output_dir,code_file_name),'a+') as f:
            for z_i in z.detach().cpu().numpy():
                f.write(str(z_i))
                f.write('\n')



        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc
                if name=='env0_validation':
                    if acc>max_validataion_acc:
                        max_validataion_acc=acc
                        report_test=True
                if name=='env0_test':
                    if report_test==True:
                        corresponding_test_acc=acc

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)
            results.update({
                'hparams': hparams,
                'args': vars(args)    
            })
            checkpoint_vals = collections.defaultdict(lambda: [])
            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')
        model_name='model_'+str(current_iter)+'.pkl'
        save_checkpoint(model_name)
    print("report test acc{}".format(corresponding_test_acc))
    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
