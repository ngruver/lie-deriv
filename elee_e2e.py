from curses import COLOR_GREEN
from logging import raiseExceptions
# from msilib.sequence import tables
from operator import mod
from posixpath import relpath
import re
import gc
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc
from sklearn import random_projection
#need to pip install ml_collections
import os
import sys
import tqdm
import wandb
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
#from timm.models.rpp_mixer2 import *
import timm.models
# from timm.models.resnet import resnet50
# from timm.models.vision_transformer import vit_small_patch16_224
from timm.data import create_dataset, create_loader, resolve_data_config
from grid_sample2 import grid_sample
import argparse
import pandas as pd
from functools import partial

def affine_transform(affineMatrices, img):
    flowgrid = F.affine_grid(affineMatrices, size = img.size(), align_corners=True)#.double()
    # uses manual grid sample implementation to be able to compute 2nd derivatives
    # img_out = F.grid_sample(img, flowgrid,padding_mode="reflection",align_corners=True)
    img_out = grid_sample(img, flowgrid)
    return img_out

def translate(img,t,axis='x'):
    """ Translates an image by a fraction of the size (sx,sy) in (0,1)"""
    #assert shift.shape == (2,)
    with torch.enable_grad():
        affineMatrices = torch.zeros(img.shape[0],2,3).to(img.device)#.double()
        affineMatrices[:,0,0] = 1
        affineMatrices[:,1,1] = 1
        if axis == 'x':
            affineMatrices[:,0,2] = t
        else:
            affineMatrices[:,1,2] = t
        return affine_transform(affineMatrices, img)

def rotate(img,angle):
    """ Rotates an image by angle"""
    affineMatrices = torch.zeros(img.shape[0],2,3).to(img.device)#.double()
    affineMatrices[:,0,0] = torch.cos(angle)
    affineMatrices[:,0,1] = torch.sin(angle)
    affineMatrices[:,1,0] = -torch.sin(angle)
    affineMatrices[:,1,1] = torch.cos(angle)
    return affine_transform(affineMatrices, img)

def shear(img,t,axis='x'):
    """ Shear an image by an amount t """
    affineMatrices = torch.zeros(img.shape[0],2,3).to(img.device)#.double()
    affineMatrices[:,0,0] = 1
    affineMatrices[:,1,1] = 1
    if axis == 'x':
        affineMatrices[:,0,1] = t
        affineMatrices[:,1,0] = 0
    else:
        affineMatrices[:,0,1] = 0
        affineMatrices[:,1,0] = t
    return affine_transform(affineMatrices, img)

def stretch(img,x,axis='x'):
    """ Stretch an image by an amount t """
    affineMatrices = torch.zeros(img.shape[0],2,3).to(img.device)#.double()
    if axis == 'x':
        affineMatrices[:,0,0] = 1 * (1+x)
    else:
        affineMatrices[:,1,1] = 1 * (1+x)
    return affine_transform(affineMatrices, img)

def saturate(img,t):
    img = img.clone()
    img *= (1 + t)
    return img

def jvp(f,x,u):
    """ Jacobian vector product Df(x)u vs typical autograd VJP vTDF(x).
        Uses two backwards passes: computes (vTDF(x))u and then derivative wrt to v to get DF(x)u"""
    with torch.enable_grad():
        y = f(x)
        v = torch.ones_like(y,requires_grad=True) # Dummy variable (could take any value)
        vJ = torch.autograd.grad(y,[x],[v],create_graph=True)
        Ju = torch.autograd.grad(vJ,[v],[u],create_graph=True)
        return Ju[0]

def translation_lie_deriv(model,inp_imgs,axis='x'):
    """ Lie derivative of model with respect to translation vector, assumes scalar output """
    #vector = vector.to(inp_imgs.device)
    shifted_model = lambda t: model(translate(inp_imgs,t,axis))
    t = torch.zeros(1,requires_grad=True,device=inp_imgs.device)
    lie_deriv = jvp(shifted_model,t,torch.ones_like(t,requires_grad=True))
    return lie_deriv

def rotation_lie_deriv(model,inp_imgs):
    """ Lie derivative of model with respect to rotation, assumes scalar output """
    rotated_model = lambda theta: model(rotate(inp_imgs,theta))
    t = torch.zeros(1,requires_grad=True,device=inp_imgs.device)
    lie_deriv = jvp(rotated_model,t,torch.ones_like(t))
    return lie_deriv

def shear_lie_deriv(model,inp_imgs,axis='x'):
    """ Lie derivative of model with respect to shear, assumes scalar output """
    sheared_model = lambda t: model(shear(inp_imgs,t,axis))
    t = torch.zeros(1,requires_grad=True,device=inp_imgs.device)
    lie_deriv = jvp(sheared_model,t,torch.ones_like(t))
    return lie_deriv

def stretch_lie_deriv(model,inp_imgs,axis='x'):
    """ Lie derivative of model with respect to stretch, assumes scalar output """
    stretched_model = lambda t: model(stretch(inp_imgs,t,axis))
    t = torch.zeros(1,requires_grad=True,device=inp_imgs.device)
    lie_deriv = jvp(stretched_model,t,torch.ones_like(t))
    return lie_deriv

def saturate_lie_deriv(model,inp_imgs):
    """ Lie derivative of model with respect to saturation, assumes scalar output """
    saturated_model = lambda t: model(saturate(inp_imgs,t))
    t = torch.zeros(1,requires_grad=True,device=inp_imgs.device)
    lie_deriv = jvp(saturated_model,t,torch.ones_like(t))
    return lie_deriv

def get_sample_efficiency(model,train_loader,test_loader):
    gc.collect()

    if torch.cuda.is_available():
        model = model.cuda()

    transformer = random_projection.GaussianRandomProjection(n_components=512)

    train = []
    for mb in tqdm.tqdm(train_loader):
        x = mb[0]
        if torch.cuda.is_available():
            x = x.cuda()
        h = model.forward_features(x).cpu().numpy()
        h = np.mean(h, axis=(-2,-1)) if np.ndim(h)>2 else h
        y = mb[1].numpy()
        train.append((h,y)) 
    train_x, train_y = zip(*train)
    train_x, train_y = np.concatenate(train_x), np.concatenate(train_y)
    train_x = transformer.fit_transform(train_x)

    test = []
    for mb in tqdm.tqdm(test_loader):
        x = mb[0]
        if torch.cuda.is_available():
            x = x.cuda()
        h = model.forward_features(x).cpu().numpy()
        h = np.mean(h, axis=(-2,-1)) if np.ndim(h)>2 else h
        y = mb[1].numpy()
        test.append((h,y)) 
    test_x, test_y = zip(*test)
    test_x, test_y = np.concatenate(test_x), np.concatenate(test_y)
    test_x = transformer.fit_transform(test_x)
    
    train_x_by_label = {label:train_x[train_y==label] for label in np.unique(train_y)}
    powers = np.unique(np.logspace(0,np.log2(len(train_loader.dataset)//100),num=10,base=2,dtype=int))

    evaluated_metrics = []
    for num_samples in tqdm.tqdm(powers):
        print(num_samples)
        num_samples = min(num_samples,min(len(train_x_by_label[label]) for label in train_x_by_label))
        
        train_x_by_label_subset = {label:train_x_by_label[label][:num_samples] for label in train_x_by_label}
        train_x_subset = np.concatenate([train_x_by_label_subset[label] for label in train_x_by_label_subset])
        train_y_subset = np.concatenate([np.full(num_samples,label) for label in train_x_by_label_subset])
        
        #train the model
        clf = LogisticRegression(solver='lbfgs',multi_class='multinomial',tol=1e-6,max_iter=10000).fit(train_x_subset,train_y_subset)
        test_pred = clf.predict(test_x)
        test_acc = np.mean(test_pred==test_y)

        evaluated_metrics.append(pd.Series({'samples_per_class': num_samples,
                                            'test_acc': test_acc.item()}))
   
    df = pd.concat(evaluated_metrics,axis=1).T
    df['auc'] = auc(df['samples_per_class'], df['test_acc'])
    return df

def eval_average_metrics_wstd(loader,metrics,max_mbs=None):
    total = len(loader) if max_mbs is None else min(max_mbs,len(loader))
    dfs = []
    with torch.no_grad():
        for idx, minibatch in tqdm.tqdm(enumerate(loader),total=total):
            dfs.append(metrics(minibatch))
            if max_mbs is not None and idx>=max_mbs:
                break
    df = pd.concat(dfs)
    return df

def numparams(model):
    return sum(p.numel() for p in model.parameters())

def get_equivariance_metrics(model,minibatch):
    x,y = minibatch
    if torch.cuda.is_available():
        model = model.cuda()
        x, y = x.cuda(), y.cuda()
    
    model = model.eval()

    model_probs = lambda x: F.softmax(model(x),dim=-1)
    model_out = model_probs(x)

    errs = {
        'trans_x_deriv': translation_lie_deriv(model_probs,x,axis='x').abs().cpu().data.numpy(),
        'trans_y_deriv': translation_lie_deriv(model_probs,x,axis='y').abs().cpu().data.numpy(),
        'rot_deriv': rotation_lie_deriv(model_probs,x).abs().cpu().data.numpy(),
        'shear_x_deriv': shear_lie_deriv(model_probs,x,axis='x').abs().cpu().data.numpy(),
        'shear_y_deriv': shear_lie_deriv(model_probs,x,axis='y').abs().cpu().data.numpy(),
        'stretch_x_deriv': stretch_lie_deriv(model_probs,x,axis='x').abs().cpu().data.numpy(),
        'stretch_y_deriv': stretch_lie_deriv(model_probs,x,axis='y').abs().cpu().data.numpy(),
        'saturate_err': saturate_lie_deriv(model_probs,x).abs().cpu().data.numpy()
    }

    yhat = model_out.argmax(dim=1)#.cpu()
    acc = (yhat==y).cpu().float().data.numpy()

    metrics = {}
    metrics['acc'] = pd.Series(acc)

    for k in errs:
        # for i in range(model_out.shape[-1]):
        #     metrics[k + str(i)] = pd.Series(errs[k][:,i])
        metrics[k + "_mean"] = pd.Series(errs[k].mean(-1))
    #print([f"{metric}: {value.shape}" for metric,value in metrics.items()])

    for shift_x in range(8):
        rolled_img = torch.roll(x,shift_x,2)
        rolled_yhat = model(rolled_img).argmax(dim=1)
        consistency = (rolled_yhat==yhat).cpu().data.numpy()
        metrics['consistency_x'+str(shift_x)] = pd.Series(consistency)
    for shift_y in range(8):
        rolled_img = torch.roll(x,shift_y,3)
        rolled_yhat = model(rolled_img).argmax(dim=1)
        consistency = (rolled_yhat==yhat).cpu().data.numpy()
        metrics['consistency_y'+str(shift_y)] = pd.Series(consistency)

    df = pd.DataFrame.from_dict(metrics)
    return df

def get_loaders(model, dataset, data_dir, batch_size, num_train, num_val, args, train_split='train', val_split='val'):

    dataset_train = create_dataset(dataset, root=data_dir, split=train_split, is_training=False, batch_size=batch_size)
    if num_train < len(dataset_train):
        dataset_train, _ = torch.utils.data.random_split(dataset_train, [num_train, len(dataset_train) - num_train],
                                                            generator=torch.Generator().manual_seed(42))

    dataset_eval = create_dataset(dataset, root=data_dir, split=val_split, is_training=False, batch_size=batch_size)
    if num_val < len(dataset_eval):
        dataset_eval, _ = torch.utils.data.random_split(dataset_eval, [num_val, len(dataset_eval) - num_val],
                                                            generator=torch.Generator().manual_seed(42))

    data_config = resolve_data_config(vars(args), model=model, verbose=True)

    print(data_config)

    train_loader = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=batch_size,
        is_training=False,
        use_prefetcher=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=1,
        distributed=False,
        crop_pct=data_config['crop_pct'],
        pin_memory=False,
        no_aug=True,
        hflip=0.,
        color_jitter=0.,
    )

    eval_loader = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=batch_size,
        is_training=False,
        use_prefetcher=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=1,
        distributed=False,
        crop_pct=data_config['crop_pct'],
        pin_memory=False,
        no_aug=True,
    )

    return train_loader, eval_loader

if __name__ == "__main__":
    # argparse
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Dataset / Model parameters
    parser.add_argument('--data_dir', metavar='DIR', default = '/imagenet',help='path to dataset')
    parser.add_argument('--dataset', '-d', metavar='NAME', default='',help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--train-split', metavar='NAME', default='train',help='dataset train split')
    parser.add_argument('--val-split', metavar='NAME', default='validation',help='dataset validation split')
    parser.add_argument('--expt', metavar='NAME', default='equivariance_metrics_10',help='experiment name')
    parser.add_argument('--modelname', metavar='NAME', default='resnet18', help='model name')
    parser.add_argument('--pretrained', metavar='NAME', default='True', help='use pretrained model')
    args = parser.parse_args()

    wandb.init(project='metamixer', config=args)

    if not os.path.exists(args.expt):
        os.makedirs(args.expt)

    print(args.modelname)

    pretrained = True if args.pretrained == 'True' else False
    model = getattr(timm.models,args.modelname)(pretrained=pretrained)
    model.eval()
    # imagenet_train_loader, imagenet_test_loader = get_loaders(model, dataset='imagenet',
    #                     data_dir='/imagenet', batch_size=8, num_train=100, num_val=100, 
    #                     args=args, train_split=args.train_split, val_split=args.val_split)

    evaluated_metrics = []
    # for key,loader in {'Imagenet_train':imagenet_train_loader, 'Imagenet_test':imagenet_test_loader}.items():
    #     metrics = eval_average_metrics_wstd(loader,partial(get_equivariance_metrics,model))
    #     metrics['dataset'] = key
    #     metrics['model'] = args.modelname
    #     metrics['params'] = numparams(model)
    #     print(np.mean(metrics['acc']))
    #     evaluated_metrics.append(metrics)

    # cifar_train_loader, cifar_test_loader = get_loaders(model, dataset='torch/cifar100',
    #                  data_dir='/scratch/nvg7279/cifar', batch_size=4, num_train=10000, num_val=5000,
    #                  args=args, train_split=args.train_split, val_split=args.val_split)

    # svhn_train_loader, svhn_test_loader = get_loaders(model, dataset='torch/svhn',
    #                  data_dir='/scratch/nvg7279/svhn', batch_size=4, num_train=10000, num_val=5000,
    #                  args=args, train_split=args.train_split, val_split='test')

    # _, retinopathy_loader = get_loaders(model, dataset='tfds/diabetic_retinopathy_detection',
    #                     data_dir='/scratch/nvg7279/tfds', batch_size=4, num_train=1e8, num_val=1e8, 
    #                     args=args, train_split='train', val_split='train')
   
    # _, histology_loader = get_loaders(model, dataset='tfds/colorectal_histology',
    #                     data_dir='/scratch/nvg7279/tfds', batch_size=4, num_train=1e8, num_val=1e8, 
    #                     args=args, train_split='train', val_split='train')

    # for key,loader in {'cifar100': cifar_test_loader, 'retinopathy':retinopathy_loader, 'histology':histology_loader}.items():
    #     metrics = eval_average_metrics_wstd(loader,partial(get_equivariance_metrics,model),max_mbs=(1000//4)+1)
    #     metrics['dataset'] = key
    #     metrics['model'] = args.modelname
    #     metrics['params'] = numparams(model)
    #     evaluated_metrics.append(metrics)

    # df = pd.concat(evaluated_metrics)

    cifar_train_loader, cifar_test_loader = get_loaders(model, dataset='torch/cifar100',
                     data_dir='/scratch/nvg7279/cifar', batch_size=4, num_train=200, num_val=100,
                     args=args, train_split=args.train_split, val_split=args.val_split)

    svhn_train_loader, svhn_test_loader = get_loaders(model, dataset='torch/svhn',
                     data_dir='/scratch/nvg7279/svhn', batch_size=4, num_train=200, num_val=100,
                     args=args, train_split=args.train_split, val_split='test')

    dfs = []
    with torch.no_grad():
        cifar_df = get_sample_efficiency(model, cifar_train_loader, cifar_test_loader)
        cifar_df['dataset'] = 'cifar100'
        cifar_df['model'] = args.modelname
        cifar_df['params'] = numparams(model)

        svhn_df = get_sample_efficiency(model, svhn_train_loader, svhn_test_loader)
        svhn_df['dataset'] = 'svhn'
        svhn_df['model'] = args.modelname
        svhn_df['params'] = numparams(model)

        # df['cifar_finetuning_auc'] = get_sample_efficiency(model, cifar_train_loader, cifar_test_loader)
        # df['svhn_finetuning_auc'] = get_sample_efficiency(model, svhn_train_loader, svhn_test_loader)
        dfs += [cifar_df, svhn_df]
    df = pd.concat(dfs)

    df.to_csv(os.path.join(args.expt, args.modelname + "_" + args.pretrained.lower() + ".csv"))
