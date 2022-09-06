from curses import COLOR_GREEN
from logging import raiseExceptions
# from msilib.sequence import tables
from operator import mod
from posixpath import relpath
import re
import gc
import torch
import torch.nn.functional as F
import torch.nn as nn
#need to pip install ml_collections
import os
import sys
import tqdm
import wandb
import numpy as np
import torch
#import matplotlib.pyplot as plt
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

def img_like(img_shape):
    bchw = (len(img_shape)==4 and img_shape[-2:]!=(1,1))
    is_square = (int(int(np.sqrt(img_shape[1]))+.5)**2 == img_shape[1])
    is_one_off_square = (int(int(np.sqrt(img_shape[1]))+.5)**2 == img_shape[1]-1)
    is_two_off_square = (int(int(np.sqrt(img_shape[1]))+.5)**2 == img_shape[1]-2)
    bnc = (len(img_shape)==3 and img_shape[1]!=1 and (is_square or is_one_off_square or is_two_off_square))
    return bchw or bnc
    
def num_tokens(img_shape):
    if (len(img_shape)==4 and img_shape[-2:]!=(1,1)): return 0
    #is_square = (int(int(np.sqrt(img_shape[1]))+.5)**2 == img_shape[1])
    is_one_off_square = (int(int(np.sqrt(img_shape[1]))+.5)**2 == img_shape[1]-1)
    is_two_off_square = (int(int(np.sqrt(img_shape[1]))+.5)**2 == img_shape[1]-2)
    return int(is_one_off_square*1 or is_two_off_square*2)


def bnc2bchw(bnc,num_tokens):
    b,n,c = bnc.shape
    h = w = int(np.sqrt(n))
    extra = bnc[:,:num_tokens,:]
    img = bnc[:,num_tokens:,:]
    return img.reshape(b,h,w,c).permute(0,3,1,2),extra

def bchw2bnc(bchw,tokens):
    b,c,h,w = bchw.shape
    n = h*w
    bnc = bchw.permute(0,2,3,1).reshape(b,n,c)
    return torch.cat([tokens,bnc],dim=1) # assumes tokens are at the start

def affine_transform(affineMatrices, img):
    assert img_like(img.shape)
    if len(img.shape) == 3:
        ntokens = num_tokens(img.shape)
        x, extra = bnc2bchw(img,ntokens)
    else:
        x = img
    flowgrid = F.affine_grid(affineMatrices, size = x.size(), align_corners=True)#.double()
    # uses manual grid sample implementation to be able to compute 2nd derivatives
    # img_out = F.grid_sample(img, flowgrid,padding_mode="reflection",align_corners=True)
    transformed = grid_sample(x, flowgrid)
    if len(img.shape) == 3:
        transformed = bchw2bnc(transformed,extra)
    return transformed

def translate(img,t,axis='x'):
    """ Translates an image by a fraction of the size (sx,sy) in (0,1)"""
    #assert shift.shape == (2,)
    
    affineMatrices = torch.zeros(x.shape[0],2,3).to(img.device)#.double()
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

def hyperbolic_rotate(img,angle):
    bs, _, w, h = img.size()
    affineMatrices = torch.zeros(img.shape[0],2,3).to(img.device)
    affineMatrices[:,0,0] = torch.cosh(angle)
    affineMatrices[:,0,1] = torch.sinh(angle)
    affineMatrices[:,1,0] = torch.sinh(angle)
    affineMatrices[:,1,1] = torch.cosh(angle)
    return affine_transform(affineMatrices, img)

def scale(img,s):
    bs, _, w, h = img.size()
    affineMatrices = torch.zeros(img.shape[0],2,3).to(img.device)
    affineMatrices[:,0,0] = 1-s
    affineMatrices[:,1,1] = 1-s
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
    """ Lie derivative of model with respect to translation vector, output can be a scalar or an image """
    #vector = vector.to(inp_imgs.device)
    if not img_like(inp_imgs.shape):
        return 0.
    def shifted_model(t):
        #print("Input shape",inp_imgs.shape)
        shifted_img = translate(inp_imgs,t,axis)
        z = model(shifted_img)
        #print("Output shape",z.shape)
        # if model produces an output image, shift it back
        if img_like(z.shape):
            z = translate(z,-t,axis)
        #print('zshape',z.shape)
        return z
    t = torch.zeros(1,requires_grad=True,device=inp_imgs.device)
    lie_deriv = jvp(shifted_model,t,torch.ones_like(t,requires_grad=True))
    # print('Liederiv shape',lie_deriv.shape)
    # print(model.__class__.__name__)
    # print('')
    return lie_deriv

def rotation_lie_deriv(model,inp_imgs):
    """ Lie derivative of model with respect to rotation, assumes scalar output """
    if not img_like(inp_imgs.shape):
        return 0.
    def rotated_model(t):
        rotated_img = rotate(inp_imgs,t)
        z = model(rotated_img)
        if img_like(z.shape):
            z = rotate(z,-t)
        return z
    t = torch.zeros(1,requires_grad=True,device=inp_imgs.device)
    lie_deriv = jvp(rotated_model,t,torch.ones_like(t))
    return lie_deriv

def hyperbolic_rotation_lie_deriv(model,inp_imgs):
    """ Lie derivative of model with respect to rotation, assumes scalar output """
    if not img_like(inp_imgs.shape):
        return 0.
    def rotated_model(t):
        rotated_img = hyperbolic_rotate(inp_imgs,t)
        z = model(rotated_img)
        if img_like(z.shape):
            z = hyperbolic_rotate(z,-t)
        return z
    t = torch.zeros(1,requires_grad=True,device=inp_imgs.device)
    lie_deriv = jvp(rotated_model,t,torch.ones_like(t))
    return lie_deriv

def scale_lie_deriv(model,inp_imgs):
    """ Lie derivative of model with respect to rotation, assumes scalar output """
    if not img_like(inp_imgs.shape):
        return 0.
    def scaled_model(t):
        scaled_img = scale(inp_imgs,t)
        z = model(scaled_img)
        if img_like(z.shape):
            z = scale(z,-t)
        return z
    t = torch.zeros(1,requires_grad=True,device=inp_imgs.device)
    lie_deriv = jvp(scaled_model,t,torch.ones_like(t))
    return lie_deriv


def shear_lie_deriv(model,inp_imgs,axis='x'):
    """ Lie derivative of model with respect to shear, assumes scalar output """
    if not img_like(inp_imgs.shape):
        return 0.
    def sheared_model(t):
        sheared_img = shear(inp_imgs,t,axis)
        z = model(sheared_img)
        if img_like(z.shape):
            z = shear(z,-t,axis)
        return z
    t = torch.zeros(1,requires_grad=True,device=inp_imgs.device)
    lie_deriv = jvp(sheared_model,t,torch.ones_like(t))
    return lie_deriv

def stretch_lie_deriv(model,inp_imgs,axis='x'):
    """ Lie derivative of model with respect to stretch, assumes scalar output """
    if not img_like(inp_imgs.shape):
        return 0.
    def stretched_model(t):
        stretched_img = stretch(inp_imgs,t,axis)
        z = model(stretched_img)
        if img_like(z.shape):
            z = stretch(z,-t,axis)
        return z
    t = torch.zeros(1,requires_grad=True,device=inp_imgs.device)
    lie_deriv = jvp(stretched_model,t,torch.ones_like(t))
    return lie_deriv

def saturate_lie_deriv(model,inp_imgs):
    """ Lie derivative of model with respect to saturation, assumes scalar output """
    if not img_like(inp_imgs.shape):
        return 0.
    def saturated_model(t):
        saturated_img = saturate(inp_imgs,t)
        z = model(saturated_img)
        if img_like(z.shape):
            z = saturate(z,-t)
        return z
    t = torch.zeros(1,requires_grad=True,device=inp_imgs.device)
    lie_deriv = jvp(saturated_model,t,torch.ones_like(t))
    return lie_deriv

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


def convert_inplace_relu_to_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ReLU())
        else:
            convert_inplace_relu_to_relu(child)

class flag:
    pass
singleton=flag
singleton.compute_lie=True
singleton.op_counter=0
singleton.fwd=True

#import pickle
# copyed_model = pickle.loads(pickle.dumps(model))
#model.attribute = list(model.attribute)  # where attribute was dict_keys
#model_clone = copy.deepcopy(model)


#TODO: record call order
#TODO: if module is called 2nd time, create a copy and store separately
# change self inplace?
def store_inputs(lie_deriv_type, self, inputs, outputs):
    if not singleton.fwd or not singleton.compute_lie: return
    #if hasattr(self,'_lie_norm_sum'): return
    with torch.no_grad():
        singleton.compute_lie=False
        if not hasattr(self,'_lie_norm_sum'):
            self._lie_norm_sum = []
            self._lie_norm_sum_sq = []
            self._num_probes = []
            self._op_counter = []
            self._fwd_counter = 0
            self._lie_deriv_output = []
        if self._fwd_counter ==len(self._lie_norm_sum):
            self._lie_norm_sum.append(0)
            self._lie_norm_sum_sq.append(0)
            self._num_probes.append(0)
            self._op_counter.append(singleton.op_counter)
            self._lie_deriv_output.append(0)
            singleton.op_counter+=1
            assert len(inputs)==1
            x, = inputs
            x = x+torch.zeros_like(x)
            self._lie_deriv_output[self._fwd_counter]=lie_deriv_type(self,x)
        
        self._fwd_counter+=1
        #print('fwd',self._fwd_counter)
        singleton.compute_lie=True
        #print("finished fwd",self)
        
        

def reset(self):
    try:
        #del self._input
        del self._lie_norm_sum
        del self._lie_norm_sum_sq
        del self._num_probes
        del self._op_counter
        del self._bwd_counter
        del self._lie_deriv_output
    except AttributeError:
        pass

def reset2(self):
    self._lie_norm_sum = [0.]*len(self._lie_norm_sum)
    self._lie_norm_sum_sq = [0.]*len(self._lie_norm_sum)
    self._num_probes = [0.]*len(self._lie_norm_sum)

def store_estimator(self, grad_input, grad_output):
    if singleton.compute_lie:
        with torch.no_grad():
            assert len(grad_output)==1
            bs = grad_output[0].shape[0]
            self._fwd_counter -=1 # reverse of forward ordering
            i = self._fwd_counter
            #if i!=0: return
            #print('bwd',self._fwd_counter)
            estimator = ((grad_output[0]*self._lie_deriv_output[i]).reshape(bs,-1).sum(-1)**2).cpu().data.numpy()
            self._lie_norm_sum[i]+= estimator
            self._lie_norm_sum_sq[i] += estimator**2
            self._num_probes[i] += 1
            #print("finished bwd",self)
            


from timm.models.vision_transformer import Attention as A1
from timm.models.vision_transformer_wconvs import Attention as A2
from timm.models.mlp_mixer import MixerBlock, Affine, SpatialGatingBlock
from timm.models.layers import PatchEmbed, Mlp, DropPath,BlurPool2d
#from timm.models.layers import FastAdaptiveAvgPool2d,AdaptiveAvgMaxPool2d
from timm.models.layers import GatherExcite,EvoNormBatch2d
from timm.models.senet import SEModule
from timm.models.efficientnet_blocks import SqueezeExcite
from timm.models.convit import MHSA,GPSA
leaflist = (A1,A2,MixerBlock,Affine,SpatialGatingBlock,PatchEmbed,Mlp,DropPath,BlurPool2d)
#leaflist += (nn.AdaptiveAvgPool2d,nn.MaxPool2d,nn.AvgPool2d)
leaflist += (GatherExcite,EvoNormBatch2d,nn.BatchNorm2d,nn.BatchNorm1d,nn.LayerNorm,nn.GroupNorm,SEModule,SqueezeExcite)
leaflist += (MHSA,GPSA)

def is_leaf(m):
    return (not hasattr(m, 'children') or not list(m.children())) or isinstance(m, leaflist)

def is_excluded(m):
    excluded_list = nn.Dropout
    return isinstance(m, excluded_list)

def selective_apply(m, fn):
    if is_leaf(m):
        if not is_excluded(m):
            fn(m)
    else:
        for c in m.children():
            selective_apply(c, fn)

# def excluded(module):
#     if isinstance(module,):
#         return False
#     if list(module.children()):
#         return True # not a leaf
#     if isinstance(module, nn.Dropout):
#         return True
#     return False

# def register_store_inputs(module):
#     module.register_forward_hook(store_inputs)
# def register_store_estimator(module):
#     module.register_backward_hook(store_estimator)
#     #module.register_backward_hook(store_estimator) #which to use?
def apply_hooks(model,lie_deriv_type):
    selective_apply(model, lambda m: m.register_forward_hook(partial(store_inputs,lie_deriv_type)))
    selective_apply(model, lambda m: m.register_backward_hook(store_estimator))


def compute_equivariance_attribution(model,img_batch,num_probes=100):
    BS = img_batch.shape[0]
    model_fn = lambda z: F.softmax(model(z),dim=1)
    all_errs = []
    order = []
    for j in range(num_probes):
        singleton.fwd=True
        y = model_fn(img_batch)
        z = torch.randn(img_batch.shape[0],1000).to(device)
        loss = (z*y).sum()
        singleton.fwd=False
        loss.backward()
        model.zero_grad()
        singleton.op_counter=0

        errs = {}
        for name, module in model.named_modules():
            if hasattr(module, '_lie_norm_sum'):
                for i in range(len(module._lie_norm_sum)):
                    assert module._num_probes[i]==1
                    
                    lie_norm = module._lie_norm_sum[i]/module._num_probes[i]
                    mod = module.__class__.__name__
                    errs[(name+(f'{i}' if i else ''),mod,module._op_counter[i],i)] = lie_norm.item()

        #errs = pd.Series(errs)
        errs = pd.Series(errs,index=pd.MultiIndex.from_tuples(errs.keys()),name=j)
        all_errs.append(errs)
        selective_apply(model,reset2)
    model.apply(reset)
    df = pd.DataFrame(all_errs)
    return df




if __name__ == "__main__":
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Dataset / Model parameters
    parser.add_argument('--data_dir', metavar='DIR', default = '/imagenet',help='path to dataset')
    parser.add_argument('--dataset', '-d', metavar='NAME', default='',help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--train-split', metavar='NAME', default='/imagenet/train',help='dataset train split')
    parser.add_argument('--val-split', metavar='NAME', default='/imagenet/val',help='dataset validation spli')
    parser.add_argument('--expt', metavar='NAME', default='layerwise_metrics',help='experiment name')
    parser.add_argument('--modelname', metavar='NAME', default='resnetblur18', help='model name')
    parser.add_argument('--num_imgs', type=int, default=20, help='Number of images to evaluate over')
    parser.add_argument('--num_probes',type=int,  default=100, help='Number of probes to use in the estimator')
    parser.add_argument('--transform', metavar='NAME', default='translation',help='translation or rotation')
    args = parser.parse_args()
    # if not os.path.exists(args.expt):
    #     os.makedirs(args.expt)
    #print(args.modelname)
    #modelname = 'resnetblur50'
    pretrained = True
    model = getattr(timm.models,args.modelname)(pretrained=pretrained)
    convert_inplace_relu_to_relu(model)
    #model = nn.Sequential(*get_children(model))
    #print(model)
    model.eval()
    device = torch.device('cuda')
    model.to(device)
    #model.to('cpu')#cuda')
    
    # selectively apply hook to model
    #model.apply(register_store_inputs)
    #model.apply(register_store_estimator)lS
    lie_deriv_type = {'translation':translation_lie_deriv,'rotation':rotation_lie_deriv,
                        'hyper_rotation':hyperbolic_rotation_lie_deriv,'scale':scale_lie_deriv,
                        'saturate':saturate_lie_deriv}[args.transform]

    apply_hooks(model,lie_deriv_type)

    BS = 1
    imagenet_train_loader, imagenet_test_loader = get_loaders(model, dataset='imagenet',
                                    data_dir='/imagenet', batch_size=BS, num_train=args.num_imgs, num_val=args.num_imgs,args=args)
    errlist = []
    for idx, (x,y) in tqdm.tqdm(enumerate(imagenet_test_loader), total=len(imagenet_test_loader)):
        if idx >= args.num_imgs:
            break 
        img = x.to(device)#x[0][None].to(device)
        errors = compute_equivariance_attribution(model,img,num_probes=args.num_probes)
        # for k,(v,err,mod,count,*rest) in errors.items():
        #     #(avg_lie_norm,avg_probe_stderr,mod,module._op_counter[i],module._num_probes[i])
        #     print(f"{k:<25} |Lf|^2 contribution: {np.sqrt(v):.4f} ± {np.sqrt(err):.4f}. {mod} {count}")
        #print(f"Sum of |Lf|^2 contributions: {sum([np.sqrt(v) for _,(v,*rest) in errors.items()]):.4f}")
        singleton.compute_lie=False
        L_all = (translation_lie_deriv(lambda z: F.softmax(model(z),dim=1),img)**2).reshape(BS,-1).sum().cpu().data.numpy()
        errors['L_all'] = L_all
        errors['img_idx'] = idx
        singleton.compute_lie=True
        #print(f"Full |L|^2: {np.sqrt(L_all):.4f}")
        errlist.append(errors)

    df =pd.concat(errlist,axis=0)
    df['model'] = args.modelname
    os.makedirs(args.expt+'_'+args.transform,exist_ok=True)
    df.to_csv(os.path.join(args.expt+'_'+args.transform, args.modelname + ".csv"))
